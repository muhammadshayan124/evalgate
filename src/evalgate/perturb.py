"""Metamorphic robustness testing: change the input in ways that should NOT change the answer, and measure whether it does.

Each perturbation preserves meaning by construction (typos at a low rate, case, whitespace, an irrelevant extra
sentence, shuffled order of a list of options). A system that is right on the original but wrong on the perturbed
version is brittle in a way a plain accuracy number hides. These are benign transformations; nothing here crafts
adversarial or harmful text.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from evalgate import stats
from evalgate.core import Case, Run

Perturbation = Callable[[str, random.Random], str]


def typos(rate: float = 0.03) -> Perturbation:
    """Swap two adjacent letters inside words with probability `rate` per word (words of 4+ letters only)."""
    def fn(text: str, rng: random.Random) -> str:
        words = text.split(" ")
        for i, w in enumerate(words):
            if len(w) >= 4 and w.isalpha() and rng.random() < rate * 4:
                j = rng.randrange(1, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        return " ".join(words)
    return fn


def casing() -> Perturbation:
    return lambda text, rng: text.upper() if rng.random() < 0.5 else text.lower()


def whitespace() -> Perturbation:
    return lambda text, rng: "  ".join(text.split(" ")).strip() + " "


def distractor(sentence: str = "Please answer as concisely as you can.") -> Perturbation:
    """Append an irrelevant, harmless sentence."""
    return lambda text, rng: f"{text} {sentence}"


def perturb_cases(cases: list[Case], fn: Perturbation, seed: int = 0) -> list[Case]:
    """Same ids, expectations and tags; only the (string) input changes. Deterministic given the seed."""
    out = []
    for c in cases:
        rng = random.Random(f"{seed}:{c.id}")
        out.append(replace(c, input=fn(c.input, rng) if isinstance(c.input, str) else c.input))
    return out


def consistency(original: Run, perturbed: Run, scorer: str, alpha: float = 0.05) -> dict:
    """How much the perturbation changes outcomes, per case and overall, with an interval on the pass-rate change."""
    if original.dataset_hash == perturbed.dataset_hash:
        raise ValueError("the perturbed run has the same dataset hash as the original; did the perturbation change anything?")
    ids = original.case_ids()
    a, b = original.passed(scorer, ids), perturbed.passed(scorer, ids)
    d = stats.paired_diff(a.astype(float), b.astype(float), alpha)
    flips_to_fail = [i for i, x, y in zip(ids, a, b, strict=True) if x and not y]
    flips_to_pass = [i for i, x, y in zip(ids, a, b, strict=True) if not x and y]
    return {
        "n": len(ids), "pass_rate_original": float(a.mean()), "pass_rate_perturbed": float(b.mean()),
        "drop": float(a.mean() - b.mean()), "drop_ci95": (float(-d.ci[1]), float(-d.ci[0])),
        "outcome_unchanged": float(np.mean(a == b)), "newly_failing": flips_to_fail, "newly_passing": flips_to_pass,
    }
