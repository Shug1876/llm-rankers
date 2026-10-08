"""Late-interaction retrieval with LightOn's PyLate models through ``pyterrier-pylate`` (PLAID index, fast-plaid).

* English sets: ``lightonai/colbertv2.0`` (the PyLate port of Stanford ColBERTv2).
* MIRACL (non-English): ``lightonai/mLateOn`` (mmBERT-base, multilingual); documents capped at 512 tokens.

Query/document lengths are the model defaults unless ``--colbert_query_length`` / ``--colbert_doc_length`` are given.
"""
import logging
from functools import lru_cache

import more_itertools
from pyterrier_pylate import PlaidIndex, PyLateBiEncoder
from tqdm import tqdm

from .common import index_path

log = logging.getLogger(__name__)

COLBERT_EN = 'lightonai/colbertv2.0'
COLBERT_ML = 'lightonai/mLateOn'
MODEL_ARGS = {COLBERT_ML: {'document_length': 512}}


@lru_cache(maxsize=None)
def _model(model_name, batch_size, query_length, doc_length):
    kwargs = dict(MODEL_ARGS.get(model_name, {}))
    if query_length:
        kwargs['query_length'] = query_length
    if doc_length:
        kwargs['document_length'] = doc_length
    return PyLateBiEncoder(model_name, batch_size=batch_size, verbose=False, **kwargs)


def _encoded_corpus(model, bench, chunk=8192):
    with tqdm(desc=f'ColBERT encode {bench.name}', unit='doc', mininterval=10) as bar:
        for docs in more_itertools.chunked(bench.corpus_iter(), chunk):
            embs = model.encode_docs([d['text'] for d in docs])
            for d, e in zip(docs, embs):
                yield {'docno': d['docno'], 'doc_embs': e}
            bar.update(len(docs))


def retriever(bench, k, args):
    model_name = args.model or (COLBERT_EN if bench.lang in ('en', 'code') else COLBERT_ML)
    model = _model(model_name, args.batch_size or 128, args.colbert_query_length, args.colbert_doc_length)
    path = index_path('colbert-' + model_name.split('/')[-1], bench.name, 'plaid')
    index = PlaidIndex(str(path), verbose=False)
    if not index.built():
        log.info('building %s', path)
        # the indexer buffers this many documents' token embeddings (fp32) before each PLAID add
        index.indexer(mode='overwrite', batch_size=args.plaid_batch).index(_encoded_corpus(model, bench))
    info = {'model': model_name, 'index': str(path),
            'query_length': model.model.query_length, 'document_length': model.model.document_length}
    return model.query_encoder() >> index.retriever(k=k), info
