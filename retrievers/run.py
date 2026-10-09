"""Index (if needed), retrieve the top k, and evaluate one retriever over one or more datasets.

    python retrievers/run.py --retriever bm25 --dataset beir/scifact
    python retrievers/run.py --retriever e5 --dataset miracl            # a whole group
    python retrievers/run.py --retriever splade --dataset bright/biology,bright/pony
    python retrievers/run.py --list                                     # every dataset name and group

Writes retrievers/experiments/<retriever>/<dataset>/{run.trec.gz, metrics.json, perquery.csv}; indexes go to
retrievers/index/<retriever-model>/<dataset>.<ext> (or the shared corpus, e.g. msmarco-passage) and are reused. A dataset whose metrics.json exists is skipped
unless --overwrite_run is given.
"""
import argparse
import importlib
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rbench import common, datasets  # noqa: E402

RETRIEVERS = ['bm25', 'splade', 'e5', 'colbert']


def run_one(retriever_name, bench, args):
    # a --model override gets its own results folder so it never overwrites the default model's runs
    label = f'{retriever_name}-{args.model.split("/")[-1]}' if args.model else retriever_name
    out = common.exp_dir(label, bench.name)
    if (out / 'metrics.json').exists() and not args.overwrite_run:
        print(f'[skip] {label} {bench.name}: {out}/metrics.json exists')
        return
    module = importlib.import_module(f'rbench.{retriever_name}')
    t0 = time.time()
    pipe, info = module.retriever(bench, args.k + bench.filter_margin, args)
    t_index = time.time() - t0

    topics = bench.topics
    t0 = time.time()
    run = pipe(topics)
    t_search = time.time() - t0
    run = run[['qid', 'docno', 'score']].astype({'qid': str, 'docno': str, 'score': 'float64'})
    run = common.cut_and_rank(bench.post_filter(run), args.k)

    out.mkdir(parents=True, exist_ok=True)
    common.save_run(run, out / 'run.trec.gz', label)
    agg, per_query, n_eval = common.evaluate(run, bench.qrels, bench.measures)
    per_query.to_csv(out / 'perquery.csv', index=False)
    common.write_json(out / 'metrics.json', {
        'dataset': bench.name, 'group': bench.group, 'retriever': label, 'k': args.k,
        'num_queries': int(topics['qid'].nunique()), 'num_eval_queries': n_eval,
        'metrics': agg, **info,
        'seconds_index_or_load': round(t_index, 1), 'seconds_search': round(t_search, 1),
    })
    scores = '  '.join(f'{m}={v:.4f}' for m, v in agg.items())
    print(f'[done] {label:7s} {bench.name:32s} {scores}  ({n_eval} queries, search {t_search:.0f}s)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--retriever', choices=RETRIEVERS)
    parser.add_argument('--dataset', help='name, group (beir, msmarco, bright, miracl, coir, all), prefix, or comma list')
    parser.add_argument('--k', type=int, default=100, help='results per query (default 100)')
    parser.add_argument('--threads', type=int, default=None, help='default: $SLURM_CPUS_PER_TASK or all cores')
    parser.add_argument('--model', default=None,
                        help='override the Hugging Face model (default: the English or multilingual one per dataset)')
    parser.add_argument('--batch_size', type=int, default=None, help='encoder batch size (model-specific default)')
    parser.add_argument('--plaid_batch', type=int, default=50_000, help='docs buffered per PLAID add (ColBERT)')
    parser.add_argument('--colbert_query_length', type=int, default=None)
    parser.add_argument('--colbert_doc_length', type=int, default=None)
    parser.add_argument('--overwrite_run', action='store_true', help='recompute runs that already have metrics.json')
    parser.add_argument('--keep_going', action='store_true', help='log a failing dataset and continue with the next')
    parser.add_argument('--list', action='store_true', help='print dataset names and groups, then exit')
    args = parser.parse_args()

    if args.list:
        for group, names in datasets.GROUPS.items():
            if group != 'all':
                print(f'{group} ({len(names)}): {" ".join(names)}')
        return
    if not args.retriever or not args.dataset:
        parser.error('--retriever and --dataset are required')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    for noisy in ('httpx', 'huggingface_hub', 'urllib3'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    args.threads = common.num_threads(args.threads)
    names = datasets.resolve(args.dataset)
    print(f'{args.retriever}: {len(names)} dataset(s), k={args.k}, threads={args.threads}')
    failed = []
    for name in names:
        try:
            run_one(args.retriever, datasets.get(name), args)
        except Exception:
            if not args.keep_going:
                raise
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f'[failed] {len(failed)}: {" ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
