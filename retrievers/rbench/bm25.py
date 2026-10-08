"""BM25 on PISA, k1=0.9 b=0.4 (the Anserini/Pyserini BEIR setting), every thread, no progress bars.
Repeated query terms count (``query_weighted=True``), as in Lucene; this matters for long queries (ArguAna).

English (BEIR, MS MARCO, BRIGHT, CoIR, MIRACL-en) uses PISA's own tokenizer, porter2 and the Terrier stopword list.
PISA's tokenizer drops every non-ASCII character, so the other MIRACL languages are tokenized here instead
(Unicode words, character bigrams for scripts written without spaces) and indexed as term counts with the
``toks_indexer``; BM25 over those counts gives exactly the scores of a text index (checked on vaswani).
"""
import collections
import logging
import unicodedata

import pyterrier as pt
import regex
from pyterrier_pisa import PisaIndex

from .common import escape_docno, index_path, unescape_docno

log = logging.getLogger(__name__)

K1, B = 0.9, 0.4

_WORD = regex.compile(r'[\p{L}\p{N}\p{M}]+')
_NO_SPACE = regex.compile(r'[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}]+')


def multilingual_tokenize(text):
    """NFKC + casefold, Unicode words; runs of CJK/Thai characters become overlapping character bigrams
    (like Lucene's CJKAnalyzer), since these scripts do not separate words with spaces."""
    text = unicodedata.normalize('NFKC', text).casefold()
    toks = []
    for word in _WORD.findall(text):
        pos = 0
        for m in _NO_SPACE.finditer(word):
            if m.start() > pos:
                toks.append(word[pos:m.start()])
            run = m.group()
            toks.extend([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)])
            pos = m.end()
        if pos < len(word):
            toks.append(word[pos:])
    return toks


def _log_progress(it, every=500_000, what='docs'):
    n = 0
    for n, x in enumerate(it, 1):
        if n % every == 0:
            log.info('indexed %d %s', n, what)
        yield x
    log.info('indexed %d %s (done)', n, what)


def retriever(bench, k, args):
    native = bench.lang in ('en', 'code')
    tag = 'bm25-pisa-porter2' if native else 'bm25-pisa-unicode'
    path = index_path(tag, bench.name, 'pisa')
    threads = args.threads
    if native:
        index = PisaIndex(str(path), text_field='text', stemmer='porter2', stops='terrier', threads=threads)
    else:
        index = PisaIndex(str(path), stemmer='none', stops='none', threads=threads)

    if not index.built():
        log.info('building %s with %d threads', path, threads)
        docs = _log_progress(bench.corpus_iter())
        if native:
            index.index({'docno': escape_docno(d['docno']), 'text': d['text']} for d in docs)
        else:
            index.toks_indexer(scale=1.0).index(
                {'docno': escape_docno(d['docno']), 'toks': dict(collections.Counter(multilingual_tokenize(d['text'])))}
                for d in docs)

    if native:
        search = index.bm25(k1=K1, b=B, num_results=k, threads=threads, verbose=False, query_weighted=True)
    else:
        search = (
            pt.apply.query_toks(lambda r: dict(collections.Counter(multilingual_tokenize(r['query']))))
            >> index.bm25(k1=K1, b=B, num_results=k, threads=threads, verbose=False, toks_scale=1.0, query_weighted=True)
        )
    pipe = search >> pt.apply.generic(lambda df: df.assign(docno=df['docno'].map(unescape_docno)), label='unescape')
    info = {'model': f'BM25 k1={K1} b={B}', 'index': str(path),
            'tokenizer': 'pisa porter2 + terrier stops' if native else 'unicode words + CJK/Thai bigrams, no stemming'}
    return pipe, info
