"""The regression gate: compare a candidate run to a baseline run under an explicit policy and decide.

Every check is reported with the numbers behind it, and the gate also states how big a regression this eval set could
have detected at all. A green gate on 20 cases is a weak statement; the report says so rather than leaving it implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field

from evalgate import stats
from evalgate.core import Run


class MetricPolicy(BaseModel):
    name: str  # a scorer name present in both runs
    margin: float = Field(0.02, ge=0, description="largest pass-rate drop we accept (non-inferiority margin)")
    min: float | None = Field(None, description="absolute floor for the candidate's pass rate")
    primary: bool = Field(False, description="the metric used for the critical-case check")


class Policy(BaseModel):
    metrics: list[MetricPolicy]
    alpha: float = 0.05
    min_cases: int = 30
    critical_tag: str = "critical"
    latency_p95_max_increase_pct: float | None = 50.0
    error_rate_max: float = 0.02
    flake_rate_max: float = 0.10
    expected_discordant_rate: float = 0.10  # used only for the minimum-detectable-difference note

    @staticmethod
    def load(path) -> Policy:
        return Policy(**yaml.safe_load(open(path, encoding="utf-8")))


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    checks: list[Check]
    regressions: dict[str, list[str]] = field(default_factory=dict)  # scorer -> case ids that went pass -> fail
    improvements: dict[str, list[str]] = field(default_factory=dict)
    mdd: float = 0.0
    n_cases: int = 0
    notes: list[str] = field(default_factory=list)


def evaluate(base: Run, cand: Run, policy: Policy) -> GateResult:
    checks: list[Check] = []
    if base.dataset_hash != cand.dataset_hash:
        checks.append(Check("same eval set", False, f"dataset hashes differ ({base.dataset_hash} vs {cand.dataset_hash}); the runs are not comparable"))
        return GateResult(False, checks)
    ids = base.case_ids()
    missing = set(ids) - set(cand.case_ids())
    if missing:
        checks.append(Check("coverage", False, f"candidate is missing {len(missing)} cases"))
        return GateResult(False, checks)
    n = len(ids)
    checks.append(Check("eval set size", n >= policy.min_cases, f"{n} cases (policy minimum {policy.min_cases})"))

    regressions, improvements = {}, {}
    primary = next((m for m in policy.metrics if m.primary), policy.metrics[0])
    for m in policy.metrics:
        if m.name not in base.scorers or m.name not in cand.scorers:
            checks.append(Check(f"metric {m.name}", False, "scorer not present in both runs"))
            continue
        b, c = base.values(m.name, ids), cand.values(m.name, ids)
        ni = stats.non_inferiority(b, c, m.margin, policy.alpha)
        bp, cp = base.passed(m.name, ids), cand.passed(m.name, ids)
        reg, imp, p = stats.mcnemar_exact(bp, cp)
        regressions[m.name] = [i for i, x, y in zip(ids, bp, cp, strict=True) if x and not y]
        improvements[m.name] = [i for i, x, y in zip(ids, bp, cp, strict=True) if not x and y]
        lo, hi = stats.wilson(int(cp.sum()), n, policy.alpha)
        checks.append(Check(f"non-inferiority: {m.name}", ni.passed,
                            f"baseline {b.mean():.3f} -> candidate {c.mean():.3f} (diff {ni.diff:+.3f}); one-sided {1 - policy.alpha:.0%} lower bound {ni.lower:+.3f} "
                            f"must be at least -{m.margin:.3f}; pass/fail flips: {reg} regressions, {imp} improvements (McNemar p={p:.3f})"))
        if m.min is not None:
            checks.append(Check(f"floor: {m.name}", bool(cp.mean() >= m.min), f"candidate pass rate {cp.mean():.3f} (Wilson {lo:.3f}-{hi:.3f}) vs floor {m.min:.3f}"))

    crit = [i for i in ids if policy.critical_tag in base.tags.get(i, [])]
    if crit:
        lost = [i for i in crit if i in regressions.get(primary.name, [])]
        checks.append(Check(f"critical cases ({len(crit)})", not lost, f"{len(lost)} critical cases regressed: {lost[:10]}" if lost else "no critical case regressed"))

    ce = cand.error_rate()
    checks.append(Check("error rate", ce <= policy.error_rate_max, f"candidate {ce:.1%} (baseline {base.error_rate():.1%}, limit {policy.error_rate_max:.1%})"))
    if policy.latency_p95_max_increase_pct is not None and base.latency_p95() > 0:
        inc = 100 * (cand.latency_p95() / base.latency_p95() - 1)
        checks.append(Check("latency p95", inc <= policy.latency_p95_max_increase_pct,
                            f"{base.latency_p95():.0f} ms -> {cand.latency_p95():.0f} ms ({inc:+.0f}%, limit +{policy.latency_p95_max_increase_pct:.0f}%)"))
    if cand.repeats > 1:
        fl = cand.flaky(primary.name)
        checks.append(Check("flakiness", fl <= policy.flake_rate_max, f"{fl:.1%} of cases disagree across {cand.repeats} repeats (limit {policy.flake_rate_max:.1%})"))

    mdd = stats.minimum_detectable_difference(n, policy.expected_discordant_rate, policy.alpha)
    notes = [f"margin {m.margin:.3f} on '{m.name}' is below what {n} cases can resolve (about {mdd:.3f}): a candidate that is truly unchanged but disagrees on "
             f"~{policy.expected_discordant_rate:.0%} of cases can fail this check by chance. Add cases, or widen the margin."
             for m in policy.metrics if 0 < m.margin < mdd * 0.6]
    return GateResult(all(c.passed for c in checks), checks, regressions, improvements, mdd, n, notes)
