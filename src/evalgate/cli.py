from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalgate import core, gate, judge, perturb, report, stats

app = typer.Typer(help="Regression evaluation for LLM applications.", no_args_is_help=True)
console = Console(highlight=False, soft_wrap=True)


def load_system(spec: str) -> core.System:
    """'package.module:name' -> the object (called with no args first if it is a factory class or function returning a System)."""
    mod, _, attr = spec.partition(":")
    if not attr:
        raise typer.BadParameter("system must look like 'package.module:name'")
    sys.path.insert(0, str(Path.cwd()))
    obj = getattr(importlib.import_module(mod), attr)
    return obj if hasattr(obj, "id") and callable(obj) and not isinstance(obj, type) else obj()


def parse_scorers(spec: str) -> list[core.Scorer]:
    out: list[core.Scorer] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "exact":
            out.append(core.exact_match())
        elif part == "contains":
            out.append(core.contains())
        elif part == "grounded":
            out.append(core.grounded())
        elif part.startswith("hit@"):
            out.append(core.hit_at_k(int(part[4:])))
        elif part.startswith("json:"):
            out.append(core.json_valid([k for k in part[5:].split("+") if k]))
        elif part.startswith("regex:"):
            out.append(core.regex(part[6:]))
        else:
            raise typer.BadParameter(f"unknown scorer {part!r} (exact, contains, grounded, hit@K, json:key1+key2, regex:PATTERN)")
    return out


@app.command("run")
def run_cmd(
    system: Annotated[str, typer.Option(help="module:name of the system under test")],
    cases: Annotated[Path, typer.Option(help="JSONL eval set")],
    scorers: Annotated[str, typer.Option(help="comma-separated scorer specs")],
    out: Annotated[Path, typer.Option(help="where to write the run JSON")],
    repeats: int = 1,
    timeout: float = 0.0,
) -> None:
    """Run a system over an eval set and save the scored run."""
    r = core.run(load_system(system), core.load_cases(cases), parse_scorers(scorers), repeats=repeats, timeout=timeout or None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.to_json(), encoding="utf-8")
    for s in r.scorers:
        console.print(f"{r.system_id} {s}: pass rate {r.passed(s).mean():.3f} over {len(r.case_ids())} cases (errors {r.error_rate():.1%})")


@app.command("gate")
def gate_cmd(
    baseline: Path,
    candidate: Path,
    policy: Annotated[Path, typer.Option(help="YAML policy")],
    markdown: Annotated[Path, typer.Option(help="write the report here (e.g. $GITHUB_STEP_SUMMARY)")] = None,
) -> None:
    """Decide whether the candidate run may replace the baseline run. Exit code 1 means no."""
    b, c = core.Run.from_json(baseline.read_text(encoding="utf-8")), core.Run.from_json(candidate.read_text(encoding="utf-8"))
    pol = gate.Policy.load(policy)
    result = gate.evaluate(b, c, pol)
    md = report.gate_markdown(result, b, c, pol)
    console.print(md)
    if markdown:
        with markdown.open("a", encoding="utf-8") as f:
            f.write(md + "\n")
    raise typer.Exit(0 if result.passed else 1)


@app.command("robustness")
def robustness_cmd(
    system: Annotated[str, typer.Option(help="module:name of the system under test")],
    cases: Annotated[Path, typer.Option()],
    scorer: Annotated[str, typer.Option(help="a single scorer spec")],
    kind: Annotated[str, typer.Option(help="typos | casing | whitespace | distractor")] = "typos",
    seed: int = 0,
) -> None:
    """Does a meaning-preserving change to the inputs change the outcomes?"""
    fns = {"typos": perturb.typos(), "casing": perturb.casing(), "whitespace": perturb.whitespace(), "distractor": perturb.distractor()}
    sc, sysm, cs = parse_scorers(scorer), load_system(system), core.load_cases(cases)
    a = core.run(sysm, cs, sc)
    b = core.run(sysm, perturb.perturb_cases(cs, fns[kind], seed), sc)
    b.dataset_hash = "perturbed-" + b.dataset_hash  # the perturbed inputs are a different eval set by design
    res = perturb.consistency(a, b, sc[0].name)
    console.print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("newly_failing", "newly_passing")})
    console.print(f"newly failing: {res['newly_failing'][:15]}")


@app.command("power")
def power_cmd(
    n: Annotated[int, typer.Option(help="number of eval cases")] = 0,
    delta: Annotated[float, typer.Option(help="regression size you want to detect, e.g. 0.03")] = 0.0,
    discordant: Annotated[float, typer.Option(help="share of cases on which two versions typically disagree")] = 0.10,
) -> None:
    """How big a regression can N cases detect, or how many cases does a regression of size DELTA need?"""
    if n:
        console.print(f"{n} cases detect a true drop of about {stats.minimum_detectable_difference(n, discordant):.1%} (80% power, alpha 5%, {discordant:.0%} discordant)")
    if delta:
        console.print(f"detecting a {delta:.1%} drop needs about {stats.cases_needed(delta, discordant)} cases")
    if not n and not delta:
        raise typer.BadParameter("give --n and/or --delta")


@app.command("calibrate")
def calibrate_cmd(judge_file: Path, truth_file: Path) -> None:
    """Agreement between a judge's PASS/FAIL labels and trusted labels. Each file: one 'pass' or 'fail' per line, same order."""
    j = [x.strip().lower() == "pass" for x in judge_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    t = [x.strip().lower() == "pass" for x in truth_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    c = judge.calibrate(j, t)
    console.print(f"n={c.n} agreement {c.agreement:.1%} kappa {c.kappa:.2f} false-pass rate {c.false_pass_rate:.1%} false-fail rate {c.false_fail_rate:.1%}")


if __name__ == "__main__":
    app()
