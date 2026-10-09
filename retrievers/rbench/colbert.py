"""Late-interaction retrieval with LightOn's PyLate models through ``pyterrier-pylate`` (PLAID index, fast-plaid).

* English sets: ``lightonai/colbertv2.0`` (the PyLate port of Stanford ColBERTv2).
* MIRACL (non-English): ``lightonai/mLateOn`` (mmBERT-base, multilingual); documents capped at 512 tokens.

Query/document lengths are the model defaults unless ``--colbert_query_length`` / ``--colbert_doc_length`` are given.
"""
import __future__
import inspect
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


def _set_centroid_batch(n):
    """Set the token chunk fast-plaid's update compares against all centroids at once (hardcoded 4096 upstream).

    Each PLAID add after the first builds several (chunk x n_centroids) fp32 matrices, and the centroid count grows with
    every add, so large corpora run out of GPU memory there (24 GB cards at ~450k docs). Smaller chunks cap that peak.
    """
    from fast_plaid.search import update
    fn = getattr(update.update_centroids, '__wrapped_original__', update.update_centroids)
    src = inspect.getsource(fn)
    if '    batch_size = 4096\n' not in src:
        raise RuntimeError('fast_plaid update_centroids changed; cannot apply --plaid_centroid_batch')
    code = compile(src.replace('    batch_size = 4096\n', f'    batch_size = {int(n)}\n'), inspect.getsourcefile(fn),
                   'exec', flags=__future__.annotations.compiler_flag, dont_inherit=True)
    ns = {}
    exec(code, update.__dict__, ns)
    ns['update_centroids'].__wrapped_original__ = fn
    update.update_centroids = ns['update_centroids']  # process_update looks it up in the module globals


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
    path = index_path('colbert-' + model_name.split('/')[-1], bench.index_name, 'plaid')
    # use_triton=False: fast-plaid's update path (every PLAID add after the first) re-clusters outlier tokens
    # with fastkmeans' Triton kernel, which hit a device-side assert on beir/climate-fever; the torch path is safe
    index = PlaidIndex(str(path), verbose=False, use_triton=False)
    if not index.built():
        log.info('building %s', path)
        _set_centroid_batch(args.plaid_centroid_batch)
        # the indexer buffers this many documents' token embeddings (fp32) before each PLAID add
        index.indexer(mode='overwrite', batch_size=args.plaid_batch).index(_encoded_corpus(model, bench))
    info = {'model': model_name, 'index': str(path),
            'query_length': model.model.query_length, 'document_length': model.model.document_length}
    return model.query_encoder() >> index.retriever(k=k), info
