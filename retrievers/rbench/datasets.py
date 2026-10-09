"""Dataset registry: one ``Bench`` per test collection, plus groups (``beir``, ``bright``, ``miracl``, ``coir``, ``all``).

Every bench yields documents as ``{'docno', 'text'}`` dicts (title and body joined), queries as a ``qid``/``query``
frame, qrels as a ``qid``/``docno``/``label`` frame, and knows the measures its benchmark reports.
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, List, Optional

import pandas as pd

BEIR_MEASURES = ['nDCG@10', 'R@100']
MSMARCO_MEASURES = ['RR@10', 'nDCG@10', 'R@100']
# TREC DL passage qrels are graded 0-3; the binary measures count grade >= 2 as relevant (TREC DL convention)
TREC_DL_MEASURES = ['nDCG@10', 'RR(rel=2)@10', 'AP(rel=2)@100', 'R(rel=2)@100']
MSMARCO_CORPUS = 'msmarco-passage'  # dev/small, DL19 and DL20 query the same 8.8M passages: one index for all three

BEIR = {  # name -> ir_datasets id (test split). Licensed sets (bioasq, signal1m, trec-news, robust04) are left out.
    'beir/arguana': 'beir/arguana',
    'beir/climate-fever': 'beir/climate-fever',
    'beir/dbpedia-entity': 'beir/dbpedia-entity/test',
    'beir/fever': 'beir/fever/test',
    'beir/fiqa': 'beir/fiqa/test',
    'beir/hotpotqa': 'beir/hotpotqa/test',
    'beir/nfcorpus': 'beir/nfcorpus/test',
    'beir/nq': 'beir/nq',
    'beir/quora': 'beir/quora/test',
    'beir/scidocs': 'beir/scidocs',
    'beir/scifact': 'beir/scifact/test',
    'beir/trec-covid': 'beir/trec-covid',
    'beir/webis-touche2020': 'beir/webis-touche2020/v2',
}
CQADUPSTACK = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex',
               'unix', 'webmasters', 'wordpress']
BEIR.update({f'beir/cqadupstack/{f}': f'beir/cqadupstack/{f}' for f in CQADUPSTACK})
BEIR_SELF_HITS = {'beir/arguana', 'beir/quora'}  # BEIR drops results where the query is its own document

BRIGHT = ['biology', 'earth-science', 'economics', 'psychology', 'robotics', 'stackoverflow', 'sustainable-living',
          'pony', 'leetcode', 'aops', 'theoremqa-theorems', 'theoremqa-questions']

MIRACL = ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'yo', 'zh']

CSN_LANGS = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
COIR = {  # name -> Hugging Face repo prefix under CoIR-Retrieval/
    'coir/apps': 'apps',
    'coir/cosqa': 'cosqa',
    'coir/synthetic-text2sql': 'synthetic-text2sql',
    'coir/codetrans-contest': 'codetrans-contest',
    'coir/codetrans-dl': 'codetrans-dl',
    'coir/stackoverflow-qa': 'stackoverflow-qa',
    'coir/codefeedback-st': 'codefeedback-st',
    'coir/codefeedback-mt': 'codefeedback-mt',
}
COIR.update({f'coir/codesearchnet/{l}': f'CodeSearchNet-{l}' for l in CSN_LANGS})
COIR.update({f'coir/codesearchnet-ccr/{l}': f'CodeSearchNet-ccr-{l}' for l in CSN_LANGS})


def _join(title, text):
    title = (title or '').strip()
    return f'{title} {text}' if title else text


@dataclass
class Bench:
    name: str
    group: str
    lang: str
    measures: List[str]
    _docs: Callable = field(repr=False)
    _queries: Callable = field(repr=False)
    _qrels: Callable = field(repr=False)
    _exclusions: Optional[Callable] = field(default=None, repr=False)  # () -> {qid: set(docno)}
    drop_self_hits: bool = False
    corpus: Optional[str] = None  # benches with the same corpus share one index; default: their own

    @property
    def index_name(self):
        return self.corpus or self.name

    def corpus_iter(self):
        seen = set()
        for doc in self._docs():
            if doc['docno'] in seen:  # a few collections repeat ids; indexers expect them unique
                continue
            seen.add(doc['docno'])
            yield doc

    @cached_property
    def topics(self):
        return self._queries()

    @cached_property
    def qrels(self):
        return self._qrels()

    @cached_property
    def exclusions(self):
        return self._exclusions() if self._exclusions else {}

    @property
    def filter_margin(self):
        """Extra results to retrieve so that the top k is still full after post-filtering."""
        margin = 1 if self.drop_self_hits else 0
        if self.exclusions:
            margin += max(len(v) for v in self.exclusions.values())
        return margin

    def post_filter(self, run):
        if self.drop_self_hits:
            run = run[run['qid'] != run['docno']]
        if self.exclusions:
            excl = self.exclusions
            keep = [docno not in excl.get(qid, ()) for qid, docno in zip(run['qid'], run['docno'])]
            run = run[keep]
        return run


# ---------------------------------------------------------------------------------------------------- ir_datasets
def _irds(irds_id, text_fields=('title', 'text')):
    import ir_datasets
    ds = ir_datasets.load(irds_id)

    def docs():
        fields = [f for f in text_fields if f in ds.docs_cls()._fields]
        for d in ds.docs_iter():
            if len(fields) == 2:
                text = _join(getattr(d, fields[0]), getattr(d, fields[1]))
            else:
                text = getattr(d, fields[0])
            yield {'docno': d.doc_id, 'text': text}

    def queries():
        return pd.DataFrame([{'qid': q.query_id, 'query': q.text} for q in ds.queries_iter()])

    def qrels():
        return pd.DataFrame([{'qid': r.query_id, 'docno': r.doc_id, 'label': int(r.relevance)}
                             for r in ds.qrels_iter()])
    return docs, queries, qrels


def _bright_exclusions(domain):
    def load():
        from datasets import load_dataset
        examples = load_dataset('xlangai/BRIGHT', 'examples', split=domain.replace('-', '_'))
        return {str(e['id']): {x for x in e['excluded_ids'] if x != 'N/A'} for e in examples}
    return load


# ---------------------------------------------------------------------------------------------------- CoIR (HF)
def _coir(repo):
    def load(part):
        from datasets import load_dataset
        return load_dataset(f'CoIR-Retrieval/{repo}-queries-corpus', split=part)

    def docs():
        for d in load('corpus'):
            yield {'docno': str(d['_id']), 'text': _join(d.get('title'), d['text'])}

    def qrels():
        from datasets import load_dataset
        q = load_dataset(f'CoIR-Retrieval/{repo}-qrels', split='test').to_pandas()
        return pd.DataFrame({'qid': q['query_id'].astype(str), 'docno': q['corpus_id'].astype(str),
                             'label': q['score'].astype(int)})

    def queries():
        test_qids = set(qrels()['qid'])  # the queries file holds every split; evaluate on test like CoIR does
        rows = [{'qid': str(q['_id']), 'query': _join(q.get('title'), q['text'])} for q in load('queries')]
        return pd.DataFrame([r for r in rows if r['qid'] in test_qids])
    return docs, queries, qrels


# ---------------------------------------------------------------------------------------------------- registry
def _build_registry():
    reg = {}
    for name, irds_id in BEIR.items():
        reg[name] = Bench(name, 'beir', 'en', BEIR_MEASURES, *_irds(irds_id), drop_self_hits=name in BEIR_SELF_HITS)
    reg['msmarco-dev'] = Bench('msmarco-dev', 'msmarco', 'en', MSMARCO_MEASURES,
                               *_irds('msmarco-passage/dev/small'), corpus=MSMARCO_CORPUS)
    for year in ('2019', '2020'):
        reg[f'msmarco-dl{year[2:]}'] = Bench(f'msmarco-dl{year[2:]}', 'msmarco', 'en', TREC_DL_MEASURES,
                                             *_irds(f'msmarco-passage/trec-dl-{year}/judged'), corpus=MSMARCO_CORPUS)
    for d in BRIGHT:
        reg[f'bright/{d}'] = Bench(f'bright/{d}', 'bright', 'en', BEIR_MEASURES, *_irds(f'bright/{d}'),
                                   _exclusions=_bright_exclusions(d))
    for lang in MIRACL:
        reg[f'miracl/{lang}'] = Bench(f'miracl/{lang}', 'miracl', lang, BEIR_MEASURES, *_irds(f'miracl/{lang}/dev'))
    for name, repo in COIR.items():
        reg[name] = Bench(name, 'coir', 'code', BEIR_MEASURES, *_coir(repo))
    return reg


REGISTRY = _build_registry()
GROUPS = {g: [n for n, b in REGISTRY.items() if b.group == g] for g in ['beir', 'msmarco', 'bright', 'miracl', 'coir']}
GROUPS['all'] = list(REGISTRY)


def resolve(spec):
    """'beir' | 'beir/scifact' | 'miracl/sw,miracl/yo' | a prefix such as 'beir/cqadupstack' -> list of names."""
    names = []
    for part in spec.split(','):
        part = part.strip()
        if part in GROUPS:
            names.extend(GROUPS[part])
        elif part in REGISTRY:
            names.append(part)
        else:
            matches = [n for n in REGISTRY if n.startswith(part + '/')]
            if not matches:
                raise SystemExit(f'unknown dataset {part!r}; groups: {sorted(GROUPS)}; e.g. {list(REGISTRY)[:5]}')
            names.extend(matches)
    return list(dict.fromkeys(names))


def get(name):
    return REGISTRY[name]
