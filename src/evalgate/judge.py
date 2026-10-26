"""LLM-as-judge scorers, and the checks that say whether a judge can be trusted.

A judge is just a function from a prompt string to a reply string, so any model (a local one, a hosted API, a stub in
tests) plugs in. Using a judge without measuring it is the common mistake: a judge is a classifier and has an accuracy.
`calibrate` measures it against human (or programmatic) labels, and `position_bias` measures the well-known tendency to
prefer whichever answer is shown first.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from evalgate import stats
from evalgate.core import Case, FnScorer, Output, Score, Scorer

JudgeFn = Callable[[str], str]


def rubric_scorer(judge: JudgeFn, rubric: str, name: str = "judge") -> Scorer:
    """Binary judge: the reply must start with PASS or FAIL. Anything else counts as a FAIL and is recorded, so a judge that
    does not follow the format is visible instead of silently scoring."""
    def fn(case: Case, out: Output) -> Score:
        prompt = (f"{rubric}\n\nQuestion: {case.input}\nReference: {case.expected}\nAnswer: {out.text}\n\n"
                  "Reply with PASS or FAIL on the first line, then one sentence of justification.")
        reply = judge(prompt).strip()
        m = re.match(r"\s*(PASS|FAIL)\b", reply, re.I)
        if not m:
            return Score(0.0, False, f"judge reply not in format: {reply[:80]!r}")
        ok = m.group(1).upper() == "PASS"
        return Score(float(ok), ok, reply[:160])
    return FnScorer(name, fn)


@dataclass(frozen=True)
class Calibration:
    n: int
    agreement: float
    kappa: float
    false_pass_rate: float  # judge says PASS, truth says FAIL: the dangerous error for a quality gate
    false_fail_rate: float
    format_failures: int


def calibrate(judge_passes: Sequence[bool], truth: Sequence[bool], format_failures: int = 0) -> Calibration:
    j, t = np.asarray(judge_passes, dtype=bool), np.asarray(truth, dtype=bool)
    if j.shape != t.shape or len(j) == 0:
        raise ValueError("need equal-length, non-empty label sequences")
    neg, pos = int((~t).sum()), int(t.sum())
    return Calibration(len(j), float(np.mean(j == t)), stats.cohens_kappa(j.astype(int), t.astype(int)),
                       float(np.sum(j & ~t) / neg) if neg else 0.0, float(np.sum(~j & t) / pos) if pos else 0.0, format_failures)


def position_bias(pair_judge: Callable[[str, str, str], str], items: Sequence[tuple[str, str, str]]) -> dict:
    """Pairwise judge consistency. For each (question, answer_1, answer_2) the judge is asked twice with the answers in both
    orders; a consistent judge picks the same ANSWER both times. `pair_judge(question, first, second)` returns 'A' or 'B'.
    Reports how often the verdict flips with the order, and how often a flip favours the first-shown answer."""
    consistent = first_biased = 0
    for q, a1, a2 in items:
        v1 = pair_judge(q, a1, a2).strip().upper()[:1]  # answer 1 shown first
        v2 = pair_judge(q, a2, a1).strip().upper()[:1]  # answer 2 shown first
        winner1 = a1 if v1 == "A" else a2
        winner2 = a2 if v2 == "A" else a1
        if winner1 == winner2:
            consistent += 1
        elif v1 == "A" and v2 == "A":
            first_biased += 1  # picked whichever was first, both times
    n = len(items)
    lo, hi = stats.wilson(n - consistent, n) if n else (0.0, 1.0)
    return {"n": n, "inconsistent_rate": (n - consistent) / n if n else 0.0, "inconsistent_ci95": (lo, hi), "always_first": first_biased}
