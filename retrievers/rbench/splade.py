"""Learned sparse retrieval stored in PISA (impact-scored, ``quantized()``), no stemming or stopwords.

* English sets: ``naver/splade-v3`` (SPLADE max pooling of log(1+relu(logits)), same model for queries and docs).
* MIRACL (non-English): ``opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1``. Documents go
  through the model (SPLADE pooling, special tokens zeroed, weights under 0.1 x the document max pruned, as on the
  model card); queries are inference-free: each query wordpiece gets its ``idf.json`` weight.

Weights are scaled by 100 and rounded to integers for PISA (the same quantisation ``pyterrier_splade`` uses).
GPU encoding shows a tqdm bar; the PISA indexing that consumes it does not.
"""
import json
import logging
from functools import lru_cache

import more_itertools
import pyterrier as pt
import torch
from tqdm import tqdm
from pyterrier_pisa import PisaIndex

from .common import escape_docno, index_path, unescape_docno

log = logging.getLogger(__name__)

SPLADE_EN = 'naver/splade-v3'
SPLADE_ML = 'opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1'
SCALE = 100.0


class SparseEncoder:
    def __init__(self, model_name, max_length=512, batch_size=64):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.model_name = model_name
        self.opensearch = model_name.startswith('opensearch-project/')
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        if self.device.type == 'cuda':
            self.model = self.model.half()
        self.id_to_token = {i: t for t, i in self.tokenizer.get_vocab().items()}
        self.special_ids = sorted(set(self.tokenizer.all_special_ids))
        if self.opensearch:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(model_name, 'idf.json')) as f:
                self.idf = json.load(f)

    @torch.no_grad()
    def encode_docs(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                return_tensors='pt', return_token_type_ids=False).to(self.device)
        logits = self.model(**inputs).logits  # (batch, seq, vocab): pool in fp16, the full tensor is large
        reps = (torch.log1p(torch.relu(logits)) * inputs['attention_mask'].unsqueeze(-1)).max(dim=1).values.float()
        if self.opensearch:
            reps[:, self.special_ids] = 0
            reps = reps * (reps > reps.max(dim=-1, keepdim=True).values * 0.1)
        return self._to_dicts(reps)

    @torch.no_grad()
    def encode_queries(self, texts):
        if self.opensearch:  # inference-free: idf weight of every distinct query token
            out = []
            for ids in self.tokenizer(texts, truncation=True, max_length=self.max_length)['input_ids']:
                toks = {self.id_to_token[i] for i in ids if i not in self.special_ids}
                out.append({t: self.idf.get(t, 0.0) * SCALE for t in toks if self.idf.get(t, 0.0) > 0})
            return out
        return [{t: float(w) for t, w in d.items()} for d in self.encode_docs(texts)]

    def _to_dicts(self, reps):
        reps = torch.round(reps * SCALE).to(torch.int32).cpu()
        rows, cols = torch.nonzero(reps, as_tuple=True)
        vals = reps[rows, cols].tolist()
        out = [{} for _ in range(reps.shape[0])]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals):
            out[r][self.id_to_token[c]] = v
        return out


@lru_cache(maxsize=None)
def _encoder(model_name, batch_size):
    return SparseEncoder(model_name, batch_size=batch_size)


def _encoded_corpus(enc, bench):
    """Yield {'docno', 'toks'} for every document, encoding in GPU batches (tqdm here only: PISA reads this
    iterator from Python and its own merge/compression steps print plain log lines)."""
    with tqdm(desc=f'SPLADE encode {bench.name}', unit='doc', mininterval=10) as bar:
        for docs in more_itertools.chunked(bench.corpus_iter(), 8192):
            # length-sorted batches waste far less padding; results are put back in corpus order
            order = sorted(range(len(docs)), key=lambda i: len(docs[i]['text']))
            toks = [None] * len(docs)
            for batch in more_itertools.chunked(order, enc.batch_size):
                for i, t in zip(batch, enc.encode_docs([docs[i]['text'] for i in batch])):
                    toks[i] = t
            for doc, t in zip(docs, toks):
                yield {'docno': escape_docno(doc['docno']), 'toks': t}
            bar.update(len(docs))


def retriever(bench, k, args):
    model_name = args.model or (SPLADE_EN if bench.lang in ('en', 'code') else SPLADE_ML)
    tag = 'splade-' + model_name.split('/')[-1]
    path = index_path(tag, bench.index_name, 'pisa')
    index = PisaIndex(str(path), stemmer='none', stops='none', threads=args.threads)
    enc = _encoder(model_name, args.batch_size or (32 if model_name == SPLADE_ML else 64))
    if not index.built():
        log.info('building %s', path)
        index.toks_indexer(scale=1.0).index(_encoded_corpus(enc, bench))

    def encode_queries(df):
        toks = []
        for batch in more_itertools.chunked(df['query'].tolist(), enc.batch_size):
            toks.extend(enc.encode_queries(batch))
        return df.assign(query_toks=toks)

    pipe = (
        pt.apply.generic(encode_queries, label='sparse query encoder')
        >> index.quantized(num_results=k, threads=args.threads, verbose=False, toks_scale=1.0)
        >> pt.apply.generic(lambda df: df.assign(docno=df['docno'].map(unescape_docno)), label='unescape')
    )
    return pipe, {'model': model_name, 'index': str(path), 'max_length': enc.max_length}
