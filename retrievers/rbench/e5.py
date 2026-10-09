"""Dense retrieval with E5 (pyterrier-dr): ``intfloat/e5-base-v2`` for English, ``intfloat/multilingual-e5-base``
for MIRACL. ``pyterrier_dr.E5`` adds the "query: " / "passage: " prefixes and L2-normalises, so dot = cosine.

Search is exact. The index is loaded onto the GPU in fp16 when it fits; otherwise (MIRACL en/de/fr/es...) numpy
brute force over the memory-mapped vectors on every CPU thread.
"""
import logging
import os
from functools import lru_cache

import more_itertools
import numpy as np
import pyterrier_dr
import torch
from tqdm import tqdm

from .common import index_path

log = logging.getLogger(__name__)

E5_EN = 'intfloat/e5-base-v2'
E5_ML = 'intfloat/multilingual-e5-base'
GPU_INDEX_BUDGET_GB = float(os.environ.get('RB_GPU_INDEX_GB', 30))


class _E5(pyterrier_dr.E5):
    """E5 run in fp16 on the GPU; vectors are cast back to fp32, the only dtype FlexIndex stores."""
    def encode_queries(self, texts, batch_size=None):
        return super().encode_queries(texts, batch_size=batch_size).astype(np.float32)

    def encode_docs(self, texts, batch_size=None):
        return super().encode_docs(texts, batch_size=batch_size).astype(np.float32)


@lru_cache(maxsize=None)
def _model(model_name, batch_size):
    model = _E5(model_name, batch_size=batch_size)
    if torch.cuda.is_available():
        model.model.half()
    return model


def _encoded_corpus(model, bench, chunk=8192):
    with tqdm(desc=f'E5 encode {bench.name}', unit='doc', mininterval=10) as bar:
        for docs in more_itertools.chunked(bench.corpus_iter(), chunk):
            vecs = model.encode_docs([d['text'] for d in docs])
            for d, v in zip(docs, vecs):
                yield {'docno': d['docno'], 'doc_vec': v}
            bar.update(len(docs))


def retriever(bench, k, args):
    model_name = args.model or (E5_EN if bench.lang in ('en', 'code') else E5_ML)
    model = _model(model_name, args.batch_size or 256)
    path = index_path('e5-' + model_name.split('/')[-1], bench.name, 'flex')
    index = pyterrier_dr.FlexIndex(str(path), verbose=False)
    if not index.built():
        log.info('building %s', path)
        index.indexer(mode='overwrite').index(_encoded_corpus(model, bench))

    num_docs = len(index)
    gpu_gb = num_docs * model.model.get_sentence_embedding_dimension() * 2 / 1e9
    if torch.cuda.is_available() and gpu_gb <= GPU_INDEX_BUDGET_GB:
        search, backend = index.torch_retriever(num_results=k, fp16=True, drop_query_vec=True), 'torch-fp16-gpu'
    else:
        search, backend = index.np_retriever(num_results=k, drop_query_vec=True), 'numpy-cpu'
    log.info('%d docs (%.1f GB fp16) -> %s search', num_docs, gpu_gb, backend)
    return model.query_encoder() >> search, {'model': model_name, 'index': str(path), 'search': f'exact ({backend})'}
