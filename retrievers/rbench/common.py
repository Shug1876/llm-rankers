"""Paths, thread count, run files and evaluation shared by every retriever."""
import gzip
import json
import os
from pathlib import Path

import ir_measures
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # retrievers/
INDEX_DIR = Path(os.environ.get('RB_INDEX_DIR', ROOT / 'index'))
EXP_DIR = Path(os.environ.get('RB_EXP_DIR', ROOT / 'experiments'))


def num_threads(requested=None):
    """--threads, else the SLURM allocation, else every core on the machine."""
    if requested:
        return int(requested)
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    return len(os.sched_getaffinity(0))


def slug(name):
    return name.replace('/', '_')


def index_path(retriever_tag, index_name, suffix):
    path = INDEX_DIR / retriever_tag / f'{slug(index_name)}.{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def exp_dir(retriever, dataset_name):
    return EXP_DIR / retriever / slug(dataset_name)


# PISA's text indexer writes "<docno> <text>" lines, so a docno must not contain whitespace (BRIGHT ids do).
# Escape before indexing and unescape on the way out; the escape is reversible and a no-op for ordinary ids.
def escape_docno(docno):
    return docno.replace('%', '%25').replace(' ', '%20').replace('\t', '%09').replace('\n', '%0A').replace('\r', '%0D')


def unescape_docno(docno):
    return docno.replace('%0D', '\r').replace('%0A', '\n').replace('%09', '\t').replace('%20', ' ').replace('%25', '%')


def cut_and_rank(run, k):
    """Sort by score within each query, keep the top k, and renumber ranks from 0."""
    run = run.sort_values(['qid', 'score'], ascending=[True, False], kind='stable')
    run = run.groupby('qid', sort=False).head(k).copy()
    run['rank'] = run.groupby('qid', sort=False).cumcount()
    return run.reset_index(drop=True)


def save_run(run, path, tag):
    with gzip.open(path, 'wt') as fout:
        for qid, docno, rank, score in zip(run['qid'], run['docno'], run['rank'], run['score']):
            fout.write(f'{qid} Q0 {docno} {int(rank) + 1} {float(score):.6f} {tag}\n')


def evaluate(run, qrels, measures):
    """Mean of each measure over every query with at least one relevant judgement.

    Queries the run does not return count as 0 (like ``trec_eval -c``), so a retriever cannot score higher by
    failing on hard queries.
    """
    measures = [ir_measures.parse_measure(m) if isinstance(m, str) else m for m in measures]
    qrels = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
    qrels = qrels[['query_id', 'doc_id', 'relevance']].astype({'query_id': str, 'doc_id': str, 'relevance': int})
    run_df = run.rename(columns={'qid': 'query_id', 'docno': 'doc_id'})[['query_id', 'doc_id', 'score']]
    eval_qids = sorted(set(qrels.loc[qrels['relevance'] > 0, 'query_id']))
    per_query = {(qid, str(m)): 0.0 for qid in eval_qids for m in measures}
    for metric in ir_measures.iter_calc(measures, qrels, run_df):
        if (metric.query_id, str(metric.measure)) in per_query:
            per_query[(metric.query_id, str(metric.measure))] = float(metric.value)
    per_query = pd.DataFrame([{'qid': q, 'measure': m, 'value': v} for (q, m), v in per_query.items()])
    agg = {str(m): float(per_query.loc[per_query['measure'] == str(m), 'value'].mean()) if eval_qids else 0.0
           for m in measures}
    return agg, per_query, len(eval_qids)


def write_json(path, obj):
    with open(path, 'wt') as fout:
        json.dump(obj, fout, indent=2)
