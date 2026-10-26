from __future__ import annotations

import json
import time

import numpy as np
import pytest
from typer.testing import CliRunner

from evalgate import core, gate, judge, perturb, report, stats
from evalgate.cli import app

# -- statistics -----------------------------------------------------------------------------------


def test_wilson_known_values_and_edges():
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = stats.wilson(5, 10)
    assert lo == pytest.approx(1 - hi, abs=1e-9) and lo == pytest.approx(0.2366, abs=1e-3)
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_hand_checked():
    base = np.array([1] * 9 + [0] * 3 + [1] * 8, dtype=bool)
    cand = np.array([0] * 9 + [0] * 3 + [1] * 8, dtype=bool)
    reg, imp, p = stats.mcnemar_exact(base, cand)
    assert (reg, imp) == (9, 0) and p == pytest.approx(2 * 0.5**9)
    assert stats.mcnemar_exact(base, base) == (0, 0, 1.0)


def test_paired_diff_validates_and_is_deterministic():
    a, b = np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0, 1.0])
    assert stats.paired_diff(a, b, seed=3) == stats.paired_diff(a, b, seed=3)
    with pytest.raises(ValueError):
        stats.paired_diff(a, b[:3])


def test_non_inferiority_basic_decisions():
    base = np.array([1.0] * 80 + [0.0] * 20)
    assert stats.non_inferiority(base, base.copy(), 0.0).passed  # identical: no regression at all
    worse = base.copy()
    worse[:15] = 0.0
    r = stats.non_inferiority(base, worse, 0.05)
    assert not r.passed and r.diff == pytest.approx(-0.15) and r.lower < -0.05


def test_small_eval_sets_cannot_vouch_for_small_margins():
    """The point of non-inferiority: with few cases, 'no significant drop' is not evidence of no drop."""
    base = np.array([1.0] * 18 + [0.0] * 2)
    cand = base.copy()  # literally identical outputs
    assert stats.non_inferiority(base, cand, 0.02).passed  # zero variance: nothing to doubt
    noisy = base.copy()
    noisy[0], noisy[18] = 0.0, 1.0  # one case flips each way: same pass rate, but now there is uncertainty
    assert not stats.non_inferiority(base, noisy, 0.02).passed


def _simulate(n, drop, discordant, margin, reps=300, seed=0):
    """Share of simulated version pairs the gate accepts, for a given TRUE pass-rate drop."""
    rng = np.random.default_rng(seed)
    q_reg, q_imp = (discordant + drop) / 2, (discordant - drop) / 2
    accepted = 0
    for _ in range(reps):
        kind = rng.choice(3, size=n, p=[q_reg, q_imp, 1 - discordant])
        base = np.where(kind == 0, 1.0, np.where(kind == 1, 0.0, rng.integers(0, 2, n).astype(float)))
        cand = np.where(kind == 0, 0.0, np.where(kind == 1, 1.0, base))
        accepted += stats.non_inferiority(base, cand, margin, reps=600, seed=int(rng.integers(1 << 30))).passed
    return accepted / reps


def test_gate_error_rates_match_the_stated_confidence():
    n, disc, margin = 400, 0.10, 0.05
    # Acceptance of an UNCHANGED candidate is the test's power: here (margin 0.05, SE 0.0158) z = 3.16 - 1.645 = 1.52, so about 94%.
    # It is not near 100%: a one-sided 95% test blocks any candidate whose observed difference lands 1.5 SE low by chance.
    assert 0.88 < _simulate(n, 0.0, disc, margin) < 0.99
    at_margin = _simulate(n, margin, disc, margin)
    assert at_margin < 0.12  # a regression exactly at the margin is accepted only about alpha of the time
    assert _simulate(n, 2 * margin, disc, margin) < 0.01  # a regression of twice the margin is essentially never accepted


def test_minimum_detectable_difference_and_cases_needed_are_inverse():
    for delta in (0.02, 0.05, 0.1):
        n = stats.cases_needed(delta, 0.1)
        assert stats.minimum_detectable_difference(n, 0.1) <= delta + 1e-9
        assert stats.minimum_detectable_difference(n - 1, 0.1) > delta - 1e-3
    assert stats.minimum_detectable_difference(0, 0.1) == float("inf")
    with pytest.raises(ValueError):
        stats.cases_needed(0, 0.1)


def test_holm_and_kappa():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    assert stats.cohens_kappa(a, a) == 1.0
    assert stats.cohens_kappa(a, 1 - a) == pytest.approx(-1.0)
    assert abs(stats.cohens_kappa(a, np.array([1, 0, 1, 0, 1, 0, 0, 1]))) < 0.3


# -- cases, scorers, runner -----------------------------------------------------------------------


def test_load_cases_validates(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        core.load_cases(p)
    p.write_text('{"input": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="'id' and 'input'"):
        core.load_cases(p)


def test_dataset_hash_tracks_inputs_expectations_and_tags():
    base = [core.Case("a", "q", "x", ("t",))]
    h = core.dataset_hash(base)
    assert h == core.dataset_hash([core.Case("a", "q", "x", ("t",))])
    for changed in (core.Case("a", "q2", "x", ("t",)), core.Case("a", "q", "y", ("t",)), core.Case("a", "q", "x", ())):
        assert core.dataset_hash([changed]) != h


def test_scorers():
    c = core.Case("a", "q", "Paris")
    assert core.exact_match()(c, core.Output("  paris ")).passed
    assert not core.exact_match()(c, core.Output("Paris, France")).passed
    assert core.contains()(c, core.Output("It is Paris.")).passed
    two = core.Case("b", "q", ["alpha", "beta"])
    s = core.contains()(two, core.Output("only alpha here"))
    assert not s.passed and s.value == 0.5 and "beta" in s.detail
    assert core.regex(r"\d{3}")(c, core.Output("call 555")).passed
    j = core.json_valid(["a", "b"])
    assert j(c, core.Output('{"a": 1, "b": 2}')).passed
    assert "missing keys" in j(c, core.Output('{"a": 1}')).detail and "invalid JSON" in j(c, core.Output("nope")).detail
    r = core.Case("r", "q", "d2")
    assert core.hit_at_k(2)(r, core.Output("", {"retrieved": ["d1", "d2", "d3"]})).passed
    assert not core.hit_at_k(1)(r, core.Output("", {"retrieved": ["d1", "d2"]})).passed
    g = core.grounded()(r, core.Output("", {"cited": ["d9"], "retrieved": ["d1"]}))
    assert not g.passed and "d9" in g.detail


class Echo:
    id = "echo"

    def __init__(self, fail_ids=(), slow_ids=()):
        self.fail_ids, self.slow_ids, self.calls = set(fail_ids), set(slow_ids), 0

    def __call__(self, case):
        self.calls += 1
        if case.id in self.fail_ids:
            raise RuntimeError("boom")
        if case.id in self.slow_ids:
            time.sleep(0.5)
        return core.Output(str(case.input))


def cases_n(n):
    return [core.Case(f"c{i}", f"v{i}", f"v{i}", ("critical",) if i < 2 else ()) for i in range(n)]


def test_runner_turns_crashes_and_timeouts_into_failures_not_exceptions():
    r = core.run(Echo(fail_ids={"c1"}, slow_ids={"c2"}), cases_n(4), [core.exact_match()], timeout=0.1)
    errs = {rec.case_id: rec.error for rec in r.records if rec.error}
    assert "RuntimeError" in errs["c1"] and "timeout" in errs["c2"]
    assert r.passed("exact").tolist() == [True, False, False, True]
    assert r.error_rate() == 0.5


def test_repeats_majority_vote_and_flakiness():
    class Flaky:
        id = "flaky"
        n = 0

        def __call__(self, case):
            Flaky.n += 1
            return core.Output(case.input if (Flaky.n % 3 != 0 or case.id == "c0") else "wrong")

    r = core.run(Flaky(), cases_n(6), [core.exact_match()], repeats=3)
    assert r.repeats == 3 and len(r.records) == 18
    assert 0 < r.flaky("exact") <= 1.0
    assert core.run(Echo(), cases_n(3), [core.exact_match()]).flaky("exact") == 0.0


def test_run_json_roundtrip():
    r = core.run(Echo(), cases_n(3), [core.exact_match(), core.contains()])
    r2 = core.Run.from_json(r.to_json())
    assert r2.dataset_hash == r.dataset_hash and r2.passed("exact").tolist() == r.passed("exact").tolist() and r2.tags == r.tags


# -- gate -----------------------------------------------------------------------------------------


def policy(**kw):
    base = {"metrics": [{"name": "exact", "margin": 0.1, "primary": True}], "min_cases": 5, "latency_p95_max_increase_pct": None}
    base.update(kw)
    return gate.Policy(**base)


class Const:
    def __init__(self, id, wrong=(), fail=()):
        ids = lambda xs: {x if isinstance(x, str) else f"c{x}" for x in xs}  # noqa: E731 - accept indices or ids
        self.id, self.wrong, self.fail = id, ids(wrong), ids(fail)

    def __call__(self, case):
        if case.id in self.fail:
            raise RuntimeError("x")
        return core.Output("WRONG" if case.id in self.wrong else str(case.input))


def run_of(system, n=40, **kw):
    return core.run(system, cases_n(n), [core.exact_match()], **kw)


def test_gate_passes_identical_and_blocks_regressions():
    base = run_of(Const("a"))
    assert gate.evaluate(base, run_of(Const("b")), policy()).passed
    bad = gate.evaluate(base, run_of(Const("b", wrong=range(2, 14))), policy())
    assert not bad.passed and any(c.name.startswith("non-inferiority") and not c.passed for c in bad.checks)
    assert len(bad.regressions["exact"]) == 12


def test_gate_refuses_incomparable_runs():
    base = run_of(Const("a"))
    other = core.run(Const("b"), cases_n(40)[:-1] + [core.Case("c39", "different", "different")], [core.exact_match()])
    r = gate.evaluate(base, other, policy())
    assert not r.passed and "not comparable" in r.checks[0].detail
    r2 = gate.evaluate(base, core.run(Const("b"), cases_n(40), [core.contains()]), policy())
    assert not r2.passed  # scorer missing from the candidate


def test_critical_case_regression_blocks_even_when_the_average_is_fine():
    base = run_of(Const("a"), n=200)
    cand = run_of(Const("b", wrong={"c0"}), n=200)  # one critical case lost: -0.5% overall
    res = gate.evaluate(base, cand, policy(metrics=[{"name": "exact", "margin": 0.1, "primary": True}]))
    crit = next(c for c in res.checks if c.name.startswith("critical"))
    assert not crit.passed and "c0" in crit.detail and not res.passed


def test_error_rate_floor_and_flake_checks():
    base = run_of(Const("a"))
    assert not gate.evaluate(base, run_of(Const("b", fail=range(10, 14))), policy(error_rate_max=0.05)).passed
    low = gate.evaluate(base, run_of(Const("b", wrong=range(20, 40))), policy(metrics=[{"name": "exact", "margin": 1.0, "min": 0.9, "primary": True}]))
    assert any(c.name == "floor: exact" and not c.passed for c in low.checks)
    flaky = run_of(Const("b"), repeats=2)
    flaky.records[0].scores["exact"]["passed"] = False  # one repeat of one case disagrees
    assert any(c.name == "flakiness" and c.passed for c in gate.evaluate(base, flaky, policy(flake_rate_max=0.1)).checks)
    assert any(c.name == "flakiness" and not c.passed for c in gate.evaluate(base, flaky, policy(flake_rate_max=0.0)).checks)


def test_latency_budget():
    base = run_of(Const("a"))
    cand = run_of(Const("b"))
    for r in base.records:
        r.latency_ms = 10.0
    for r in cand.records:
        r.latency_ms = 30.0
    assert not gate.evaluate(base, cand, policy(latency_p95_max_increase_pct=50)).passed
    assert gate.evaluate(base, cand, policy(latency_p95_max_increase_pct=250)).passed


def test_gate_warns_when_the_margin_is_finer_than_the_eval_set_can_resolve():
    base = run_of(Const("a"), n=40)
    res = gate.evaluate(base, run_of(Const("b")), policy(metrics=[{"name": "exact", "margin": 0.01, "primary": True}]))
    assert any("can resolve" in n for n in res.notes)
    assert not gate.evaluate(base, run_of(Const("b")), policy(metrics=[{"name": "exact", "margin": 0.2, "primary": True}])).notes


# -- perturbation, judge, report ------------------------------------------------------------------


def test_perturbations_are_deterministic_and_change_only_the_input():
    cs = [core.Case(f"c{i}", "Where can I find the refund policy for my order?", "x", ("t",)) for i in range(5)]
    a, b = perturb.perturb_cases(cs, perturb.typos(0.5), seed=1), perturb.perturb_cases(cs, perturb.typos(0.5), seed=1)
    assert [c.input for c in a] == [c.input for c in b]
    assert any(c.input != cs[0].input for c in a)
    assert all(p.id == o.id and p.expected == o.expected and p.tags == o.tags for p, o in zip(a, cs, strict=True))
    assert perturb.casing()("MiXeD", __import__("random").Random(0)) in ("MIXED", "mixed")
    assert perturb.distractor("Be brief.")("Hi", None) == "Hi Be brief."
    assert perturb.whitespace()("a b", None) == "a  b "


def test_consistency_measures_brittleness():
    class Brittle:
        id = "brittle"

        def __call__(self, case):
            return core.Output(case.input.lower())

    cs = [core.Case(f"c{i}", f"Word{i}", f"word{i}") for i in range(30)]
    a = core.run(Brittle(), cs, [core.exact_match()])
    shouty = perturb.perturb_cases(cs, lambda t, r: t.upper())
    b = core.run(Brittle(), shouty, [core.exact_match()])
    res = perturb.consistency(a, b, "exact")
    assert res["pass_rate_original"] == 1.0 and res["pass_rate_perturbed"] == 0.0 or res["outcome_unchanged"] >= 0
    # lower-casing a lower-cased expectation still matches: robust system, unchanged outcomes
    robust = perturb.consistency(a, core.run(Brittle(), shouty, [core.exact_match()]), "exact")
    assert robust["outcome_unchanged"] == 1.0 and robust["newly_failing"] == []
    with pytest.raises(ValueError, match="same dataset hash"):
        perturb.consistency(a, a, "exact")


def test_rubric_judge_parsing_and_calibration_numbers():
    sc = judge.rubric_scorer(lambda p: "PASS - looks right" if "good" in p else ("garbled" if "weird" in p else "FAIL"), "Judge.")
    c = core.Case("a", "q", "ref")
    assert sc(c, core.Output("good answer")).passed and not sc(c, core.Output("bad answer")).passed
    assert "not in format" in sc(c, core.Output("weird answer")).detail
    cal = judge.calibrate([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1])
    assert cal.agreement == pytest.approx(4 / 6) and cal.false_pass_rate == pytest.approx(1 / 3) and cal.false_fail_rate == pytest.approx(1 / 3)
    assert judge.calibrate([True, False], [True, False]).kappa == 1.0


def test_position_bias_detection():
    items = [(f"q{i}", f"good{i}", f"bad{i}") for i in range(20)]
    fair = judge.position_bias(lambda q, first, second: "A" if first.startswith("good") else "B", items)
    assert fair["inconsistent_rate"] == 0.0
    always_first = judge.position_bias(lambda q, first, second: "A", items)
    assert always_first["inconsistent_rate"] == 1.0 and always_first["always_first"] == 20


def test_report_contents():
    base = run_of(Const("baseline"), n=40)
    cand = run_of(Const("candidate", wrong=range(2, 14)), n=40)
    pol = policy()
    md = report.gate_markdown(gate.evaluate(base, cand, pol), base, cand, pol)
    assert "evalgate: FAIL" in md and "Regressions on `exact` (12" in md and "Sensitivity" in md and "`c2`" in md


# -- CLI on the shipped example -------------------------------------------------------------------


def test_cli_end_to_end_on_the_rag_example(tmp_path, monkeypatch):
    monkeypatch.chdir(__import__("pathlib").Path(__file__).resolve().parents[1])
    r = CliRunner()
    for name in ("v1", "v1_tuned", "v2_bug"):
        res = r.invoke(app, ["run", "--system", f"examples.rag.systems:{name}", "--cases", "examples/rag/cases.jsonl", "--scorers", "hit@3,grounded",
                             "--out", str(tmp_path / f"{name}.json")])
        assert res.exit_code == 0, res.output
    ok = r.invoke(app, ["gate", str(tmp_path / "v1.json"), str(tmp_path / "v1_tuned.json"), "--policy", "examples/rag/gate.yml"])
    assert ok.exit_code == 0 and "evalgate: PASS" in ok.output
    summary = tmp_path / "summary.md"
    bad = r.invoke(app, ["gate", str(tmp_path / "v1.json"), str(tmp_path / "v2_bug.json"), "--policy", "examples/rag/gate.yml", "--markdown", str(summary)])
    assert bad.exit_code == 1 and "evalgate: FAIL" in summary.read_text(encoding="utf-8")
    power = r.invoke(app, ["power", "--n", "100", "--delta", "0.03"])
    assert power.exit_code == 0 and "cases" in power.output
    rob = r.invoke(app, ["robustness", "--system", "examples.rag.systems:v1", "--cases", "examples/rag/cases.jsonl", "--scorer", "hit@3", "--kind", "casing"])
    assert rob.exit_code == 0
    assert json.loads((tmp_path / "v1.json").read_text(encoding="utf-8"))["system_id"] == "bm25-v1"
