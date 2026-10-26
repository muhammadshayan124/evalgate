"""Markdown for CI logs and GitHub step summaries."""

from __future__ import annotations

from evalgate.core import Run
from evalgate.gate import GateResult, Policy


def gate_markdown(result: GateResult, base: Run, cand: Run, policy: Policy, max_flips: int = 15) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    lines = [f"## evalgate: {verdict}", "", f"baseline `{base.system_id}` vs candidate `{cand.system_id}` on {result.n_cases} cases "
             f"(eval set `{base.dataset_hash}`).", "", "| check | result | detail |", "|---|---|---|"]
    for c in result.checks:
        lines.append(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {c.detail} |")
    if result.n_cases:
        lines += ["", f"_Sensitivity: with {result.n_cases} cases and about {policy.expected_discordant_rate:.0%} of cases changing between versions, this eval set "
                  f"detects a true pass-rate drop of about **{result.mdd:.1%}** with 80% power. A smaller regression can pass._"]
    for note in result.notes:
        lines += ["", f"> **Note:** {note}"]
    for name, ids in result.regressions.items():
        if ids:
            lines += ["", f"### Regressions on `{name}` ({len(ids)}: passed before, fails now)", ", ".join(f"`{i}`" for i in ids[:max_flips]) + (" ..." if len(ids) > max_flips else "")]
            by = {}
            for r in cand.records:
                by.setdefault(r.case_id, r)
            for i in ids[:5]:
                r = by[i]
                lines.append(f"- `{i}`: {r.scores.get(name, {}).get('detail') or r.error or 'failed the scorer'}")
    for name, ids in result.improvements.items():
        if ids:
            lines += ["", f"### Improvements on `{name}` ({len(ids)})", ", ".join(f"`{i}`" for i in ids[:max_flips]) + (" ..." if len(ids) > max_flips else "")]
    return "\n".join(lines)
