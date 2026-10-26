"""The statistics behind every decision the gate makes.

Eval sets are small and the comparisons are paired (the same cases, two systems), so:

* intervals on a single pass-rate use Wilson, not the normal approximation, which fails near 0 and 1;
* a change between two systems is judged on the PER-CASE differences, resampling cases (a paired bootstrap),
  because most cases are answered the same way by both and only the discordant ones carry information;
* "no regression" is a NON-INFERIORITY claim, tested one-sided against a stated margin. Saying "the difference is not
  significant" is not evidence of no regression; with 100 cases a 10-point drop is routinely "not significant".
* `minimum_detectable_difference` says how big a regression this eval set could even notice, so a green gate is read
  together with how much it could have caught.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(base_ok: np.ndarray, cand_ok: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired pass/fail. Returns (regressions, improvements, p).
    A regression is a case the baseline passed and the candidate failed."""
    b, c = np.asarray(base_ok, dtype=bool), np.asarray(cand_ok, dtype=bool)
    reg, imp = int(np.sum(b & ~c)), int(np.sum(~b & c))
    if reg + imp == 0:
        return reg, imp, 1.0
    return reg, imp, float(min(1.0, stats.binomtest(reg, reg + imp, 0.5).pvalue))


@dataclass(frozen=True)
class PairedDiff:
    n: int
    base_mean: float
    cand_mean: float
    diff: float  # candidate - baseline; negative is a regression
    lower: float  # lower bound of the one-sided (1 - alpha) interval
    upper: float  # upper bound of the one-sided (1 - alpha) interval
    ci: tuple[float, float]  # two-sided (1 - alpha) percentile interval


def paired_diff(base: np.ndarray, cand: np.ndarray, alpha: float = 0.05, reps: int = 5000, seed: int = 0) -> PairedDiff:
    base, cand = np.asarray(base, dtype=float), np.asarray(cand, dtype=float)
    if base.shape != cand.shape or base.ndim != 1:
        raise ValueError("paired scores must be 1-D arrays of the same length (same cases, same order)")
    d = cand - base
    n = len(d)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, (reps, n))].mean(axis=1) if n else np.zeros(reps)
    return PairedDiff(n, float(base.mean()) if n else 0.0, float(cand.mean()) if n else 0.0, float(d.mean()) if n else 0.0,
                      float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha))),
                      (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))))


@dataclass(frozen=True)
class NonInferiority:
    passed: bool
    diff: float
    lower: float
    margin: float
    n: int


def non_inferiority(base: np.ndarray, cand: np.ndarray, margin: float, alpha: float = 0.05, reps: int = 5000, seed: int = 0) -> NonInferiority:
    """Pass only if we can say, with one-sided confidence 1 - alpha, that the candidate is no more than `margin` worse.
    H0: candidate - baseline < -margin. The gate FAILS when the lower confidence bound falls below -margin, so an
    eval set too small to rule out a regression of that size fails the gate instead of waving it through."""
    r = paired_diff(base, cand, alpha, reps, seed)
    return NonInferiority(bool(r.lower >= -margin), r.diff, r.lower, margin, r.n)


def minimum_detectable_difference(n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true pass-rate drop this eval set detects with the given power, using the normal approximation for
    paired differences. `discordant_rate` is the share of cases on which two systems disagree (measure it, or assume
    0.1-0.2 for two similar prompts)."""
    if n <= 0:
        return float("inf")
    return float((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * math.sqrt(discordant_rate / n))


def cases_needed(delta: float, discordant_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Eval-set size needed to detect a drop of `delta` (inverse of minimum_detectable_difference)."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    return math.ceil(discordant_rate * ((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / delta) ** 2)


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Chance-corrected agreement between two raters (e.g. an LLM judge and human labels) over categorical labels."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("need two equal-length, non-empty label arrays")
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
