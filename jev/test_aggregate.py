"""Tests for the score aggregations in aggregate_pointwise.py on hand-checked grade distributions.
Run:  python jev/test_aggregate.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_pointwise import grades, score  # noqa: E402
from jev_rankers import JevPointwiseLlmRanker  # noqa: E402

# P(0), P(1), P(2), P(3)
EX = {'A': [.05, .15, .30, .50],   # probably perfect, unsure
      'B': [0, .05, .90, .05],     # confidently highly relevant
      'C': [.90, .05, .03, .02],   # confidently irrelevant
      'D': [.40, .10, .10, .40],   # split between 0 and 3
      'X': [.2, .3, .5, 0],        # mostly 2, rest on 1
      'Y': [.5, .3, .2, 0],        # X mirrored
      'E': [.5, 0, .5, 0]}         # same P(>=2) as X, rest on 0
# expected scores (2 decimals) and orders, '=' marks a tie
EXPECTED = {
    ('egrade', None): ({'A': 2.25, 'B': 2.00, 'C': 0.17, 'D': 1.50, 'X': 1.30, 'Y': 0.70, 'E': 1.00}, 'A>B>D>X>E>Y>C'),
    ('wb_ratio', None): ({'A': 1.00, 'B': 9.00, 'C': 9.00, 'D': 0.67, 'X': 1.00, 'Y': 1.00, 'E': 1.00}, 'B=C>A=E=X=Y>D'),
    ('exp2', None): ({'A': 4.55, 'B': 3.10, 'C': 0.28, 'D': 3.20, 'X': 1.80, 'Y': 0.90, 'E': 1.50}, 'A>D>B>X>E>Y>C'),
    ('logit2', None): ({'A': 1.39, 'B': 2.94, 'C': -2.94, 'D': 0.00, 'X': 0.00, 'Y': -1.39, 'E': 0.00}, 'B>A>D=E=X>Y>C'),
    ('cumlogit', None): ({'A': 4.33, 'B': 4.60, 'C': -9.03, 'D': 0.00, 'X': -3.21, 'Y': -5.98, 'E': -4.60},
                         'B>A>D>X>E>Y>C'),
    ('gain_b', 1.5): ({'A': 3.27, 'B': 2.54, 'C': 0.22, 'D': 2.25, 'X': 1.55, 'Y': 0.80, 'E': 1.25}, 'A>B>D>X>E>Y>C'),
    ('meansd', 0.5): ({'A': 1.81, 'B': 1.84, 'C': -0.11, 'D': 0.82, 'X': 0.91, 'Y': 0.31, 'E': 0.50}, 'B>A>X>D>E>Y>C'),
    ('meansd', -0.5): ({'A': 2.69, 'B': 2.16, 'C': 0.45, 'D': 2.18, 'X': 1.69, 'Y': 1.09, 'E': 1.50}, 'A>D>B>X>E>Y>C'),
}


def order(s):
    groups = {}
    for k, v in s.items():
        groups.setdefault(round(v, 6), []).append(k)
    return '>'.join('='.join(sorted(groups[v])) for v in sorted(groups, reverse=True))


def main():
    failures = 0
    for (method, param), (want, want_order) in EXPECTED.items():
        got = {k: score(method, p, param) for k, p in EX.items()}
        bad = {k: round(v, 2) for k, v in got.items() if abs(round(v, 2) - want[k]) > 1e-9}
        if bad or order(got) != want_order:
            failures += 1
            print(f'FAIL {method} {param}: wrong {bad}, order {order(got)} != {want_order}')
    for b, same in ((1.0, 'egrade'), (2.0, 'exp2')):  # gain_b reduces to the fixed gains
        failures += any(abs(score('gain_b', p, b) - score(same, p)) > 1e-9 for p in EX.values())
    failures += any(abs(score('meansd', p, 0.0) - score('egrade', p)) > 1e-9 for p in EX.values())

    score4 = {str(i): i for i in range(4)}
    p, out = grades({'0': .1, '1': .2, '2': .3, '3': .2, '4': .1, '5': .1}, score4)  # local 0-5 answer, 4-level prompt
    failures += abs(out - 0.2) > 1e-9 or any(abs(a - b) > 1e-9 for a, b in zip(p, [.125, .25, .375, .25]))
    p, _ = grades({'exact': .7, 'partial': .2, 'related': .1, 'irrelevant': 0},
                  {'exact': 3, 'partial': 2, 'related': 1, 'irrelevant': 0})
    failures += p != [0, .1, .2, .7]
    for method in ('wb_ratio', 'logit2', 'cumlogit'):  # rounded probabilities of exactly 0 / 1 must not divide by zero
        for p in ([0, 0, 0, 1.0], [1.0, 0, 0, 0], [0, 0, 1.0, 0]):
            failures += not abs(score(method, p)) < float('inf')
    r = JevPointwiseLlmRanker.__new__(JevPointwiseLlmRanker)  # the ranker's own score: 0-3 scale, no client needed
    r.method = 'score'
    local = {'score': 2.1, 'probabilities': {'0': .1, '1': .2, '2': .3, '3': .2, '4': .1, '5': .1}}
    failures += abs(r._graded(local) - 1.75) > 1e-9                       # (0*.1 + .2 + .6 + .6) / .8
    failures += r._graded({'score': 2.02, 'probabilities': {'0': 0, '1': .06, '2': .86, '3': .08}}) != 2.02  # hosted kept
    print('all tests passed' if not failures else f'{failures} failure(s)')
    sys.exit(bool(failures))


if __name__ == '__main__':
    main()
