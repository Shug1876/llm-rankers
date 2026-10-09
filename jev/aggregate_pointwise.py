"""Re-score graded pointwise runs from their answer distributions (the `.probs.jsonl` files run_jev.py writes next to
score / cookbook_score / trec / umbrela / grade4 runs), with other ways of turning P(grade) into a ranking score.

    python jev/aggregate_pointwise.py jev/runs/local9b_probs/dl*.probs.jsonl            # tune on one year, test on the other
    python jev/aggregate_pointwise.py jev/runs/local9b_probs/dl19.*.probs.jsonl --tune none

Each distribution is mapped to grades 0-3 with the row's `levels` (answer key -> grade); probability on levels the prompt
never defined (the local model's 4 and 5) is dropped and the rest renormalised. Methods (g = grade):
    raw       the score in the run file (runs made before the 0-3 fix in JevPointwiseLlmRanker._graded include 4-5)
    egrade    E[g]                                   expected grade, the current method on the cleaned distribution
    wb_ratio  P(argmax) / (1 - P(argmax))            whiteboard idea 1 (confidence, not relevance: kept for reference)
    exp2      E[2^g - 1]                             whiteboard idea 2 (expected exponential gain)
    logit2    logit P(g >= 2)                        relevant vs not, split at the TREC binary threshold
    cumlogit  sum_k logit P(g >= k), k = 1..3        all three splits
    gain_b    E[(b^g - 1) / (b - 1)]                 b tuned (b = 1: E[g], b = 2: exp2)
    meansd    E[g] - lam * SD[g]                     lam tuned (> 0 prefers certain passages, < 0 passages with upside)
Probabilities are clipped to [--clip, 1 - --clip] inside logits and ratios. Ties are broken by the BM25 rank. Each method
is written as <run>.agg-<method>.txt (TREC format) and evaluated: nDCG@10, RR(rel=2)@10, AP(rel=2)@100, and a paired
t-test of per-query nDCG@10 against egrade (Holm-corrected over the methods of one run).
"""
import argparse
import collections
import glob
import json
import math
import os

import ir_datasets
import ir_measures

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {'dl19': 'msmarco-passage/trec-dl-2019/judged', 'dl20': 'msmarco-passage/trec-dl-2020/judged'}
MEASURES = [ir_measures.parse_measure(m) for m in ('nDCG@10', 'RR(rel=2)@10', 'AP(rel=2)@100')]
NDCG = MEASURES[0]
B_GRID = [1 + 0.25 * i for i in range(13)]        # 1.0 .. 4.0
LAM_GRID = [-1 + 0.25 * i for i in range(9)]      # -1.0 .. 1.0


# ------------------------------------------------------------------------------------------------ scores
def grades(probs, levels):
    """Answer distribution -> [P(0), P(1), P(2), P(3)] over the prompt's scale, and the mass dropped outside it."""
    p = [0.0] * (max(levels.values()) + 1)
    for k, v in probs.items():
        if k in levels:
            p[levels[k]] += float(v)
    total = sum(p)
    return ([x / total for x in p] if total > 0 else [1.0 / len(p)] * len(p)), 1.0 - total


def expect(p, gain):
    return sum(gain(g) * x for g, x in enumerate(p))


def _logit(x, eps):
    x = min(max(x, eps), 1 - eps)
    return math.log(x / (1 - x))


def _gain_b(b):
    return (lambda g: g) if abs(b - 1) < 1e-9 else (lambda g: (b ** g - 1) / (b - 1))


def score(method, p, param=None, eps=0.01):
    if method == 'egrade':
        return expect(p, lambda g: g)
    if method == 'wb_ratio':
        top = max(g for g, x in enumerate(p) if x == max(p))  # ties -> the higher grade
        return p[top] / max(1 - p[top], eps)
    if method == 'exp2':
        return expect(p, lambda g: 2 ** g - 1)
    if method == 'logit2':
        return _logit(sum(p[2:]), eps)
    if method == 'cumlogit':
        return sum(_logit(sum(p[k:]), eps) for k in range(1, len(p)))
    if method == 'gain_b':
        return expect(p, _gain_b(param))
    if method == 'meansd':
        m = expect(p, lambda g: g)
        return m - param * math.sqrt(max(expect(p, lambda g: g * g) - m * m, 0.0))
    raise ValueError(method)


FIXED = ['egrade', 'wb_ratio', 'exp2', 'logit2', 'cumlogit']
TUNED = {'gain_b': B_GRID, 'meansd': LAM_GRID}


# ------------------------------------------------------------------------------------------------ runs
def dataset_of(path):
    ds = os.path.basename(path).split('.')[0]
    if ds not in DATASETS:
        raise ValueError(f'{path}: file name must start with one of {", ".join(DATASETS)}')
    return ds


def load(path, bm25_dir):
    """qid -> list of (docid, raw score, grade distribution, BM25 rank); and the mean mass outside the scale."""
    bm25 = {}
    with open(os.path.join(bm25_dir, f'run.rank_llm.bm25.{dataset_of(path)}.top100.txt')) as f:
        for line in f:
            parts = line.split()
            bm25[(parts[0], parts[2])] = int(parts[3])
    per_query, dropped = collections.defaultdict(list), []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p, out = grades(r['probs'], r['levels'])
            dropped.append(out)
            per_query[r['qid']].append((r['docid'], r['score'], p, bm25.get((r['qid'], r['docid']), 10 ** 9)))
    return per_query, sum(dropped) / len(dropped)


def rank(per_query, fn):
    """qid -> [(docid, score)] sorted by score, ties by BM25 rank."""
    return {qid: [(d, s) for s, _, d in sorted(((fn(raw, p), r, d) for d, raw, p, r in docs),
                                                key=lambda t: (-t[0], t[1]))]
            for qid, docs in per_query.items()}


def to_run(ranked):
    return [ir_measures.ScoredDoc(qid, d, len(docs) - i) for qid, docs in ranked.items() for i, (d, _) in enumerate(docs)]


def write(ranked, path, tag):
    with open(path, 'w') as f:
        for qid, docs in ranked.items():
            for i, (d, s) in enumerate(docs, 1):
                f.write(f'{qid}\tQ0\t{d}\t{i}\t{s}\t{tag}\n')


_QRELS = {}


def qrels(ds):
    if ds not in _QRELS:
        _QRELS[ds] = list(ir_datasets.load(DATASETS[ds]).qrels_iter())
    return _QRELS[ds]


def per_query(ranked, ds, measures=MEASURES):
    """measure -> {qid: value} over every judged query (missing queries count 0, as trec_eval -c)."""
    q = qrels(ds)
    out = {m: {qid: 0.0 for qid in {r.query_id for r in q}} for m in measures}
    for m in ir_measures.iter_calc(measures, q, to_run(ranked)):
        out[m.measure][m.query_id] = m.value
    return out


def mean(values):
    return sum(values.values()) / len(values)


def tune(per_q, ds, method):
    """Grid value with the best mean nDCG@10 on (per_q, ds), and the whole grid."""
    grid = {v: mean(per_query(rank(per_q, lambda raw, p: score(method, p, v)), ds, [NDCG])[NDCG])
            for v in TUNED[method]}
    return max(grid, key=grid.get), grid


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = running
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('probs', nargs='+', help='.probs.jsonl files (globs allowed); names start with dl19 / dl20')
    parser.add_argument('--tune', default='cross', choices=['cross', 'none'],
                        help='cross: pick b / lam on the other year (same file name otherwise); none: skip tuned methods')
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--bm25_dir', default=os.path.join(HERE, 'runs', 'bm25'))
    args = parser.parse_args()
    from scipy.stats import ttest_rel

    paths = [p for pattern in args.probs for p in sorted(glob.glob(pattern))] or args.probs
    loaded = {p: load(p, args.bm25_dir) for p in paths}
    summary = {}
    for path in paths:
        ds, (per_q, dropped) = dataset_of(path), loaded[path]
        base = path[:-len('.probs.jsonl')]
        name = os.path.basename(base)
        fns = {'raw': lambda raw, p: raw}
        fns.update({m: (lambda m: lambda raw, p: score(m, p, eps=args.clip))(m) for m in FIXED})
        params = {}
        if args.tune == 'cross':
            other = os.path.join(os.path.dirname(path), name.replace(ds, next(d for d in DATASETS if d != ds), 1)
                                 + '.probs.jsonl')
            if os.path.exists(other) and other in loaded:
                for m in TUNED:
                    params[m], grid = tune(loaded[other][0], dataset_of(other), m)
                    fns[m] = (lambda m, v: lambda raw, p: score(m, p, v, args.clip))(m, params[m])
            else:
                print(f'{name}: no {os.path.basename(other)} among the inputs, tuned methods skipped')

        print(f'\n{name}  ({len(per_q)} queries, mean probability on undefined levels {dropped:.4f}'
              + ''.join(f', {m}={v:g} (tuned on the other year)' for m, v in params.items()) + ')')
        results = {}
        for m, fn in fns.items():
            ranked = rank(per_q, fn)
            write(ranked, f'{base}.agg-{m}.txt', f'agg-{m}')
            results[m] = per_query(ranked, ds)
        ref = results['egrade'][NDCG]
        qids = sorted(ref)
        pv = {m: ttest_rel([r[NDCG][q] for q in qids], [ref[q] for q in qids]).pvalue
              for m, r in results.items() if m != 'egrade'}
        pv = holm({m: (1.0 if math.isnan(v) else v) for m, v in pv.items()})
        print(f'  {"method":10s}' + ''.join(f'{str(m):>15s}' for m in MEASURES) + '   p vs egrade (Holm)')
        for m, r in results.items():
            print(f'  {m:10s}' + ''.join(f'{mean(r[x]):15.4f}' for x in MEASURES)
                  + (f'   {pv[m]:.3f}' if m in pv else '   -'))
        summary[name] = {'undefined_mass': dropped, 'params': params,
                         'means': {m: {str(x): mean(r[x]) for x in MEASURES} for m, r in results.items()},
                         'p_vs_egrade_holm': pv}
    out = os.path.join(os.path.dirname(paths[0]), 'aggregate_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nruns written next to the inputs as *.agg-<method>.txt, summary in {out}')


if __name__ == '__main__':
    main()
