"""Cases, scorers and the runner.

A *case* is an input with an optional expected answer and tags (`critical` is special: see gate.py). A *scorer* turns
(case, system output) into a Score in [0, 1] with an explicit pass/fail. A *system* is any callable from a case to
an Output; it can wrap a prompt, a RAG pipeline, an HTTP endpoint or a local model, which is the point: the harness
knows nothing about what is being evaluated.

Runs are plain JSON, so a baseline run can be committed or cached as a CI artifact and a candidate compared to it later.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# -- cases ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    expected: Any = None
    tags: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


def load_cases(path: Path) -> list[Case]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if "id" not in d or "input" not in d:
            raise ValueError(f"{path}:{n}: each case needs 'id' and 'input'")
        cases.append(Case(str(d["id"]), d["input"], d.get("expected"), tuple(d.get("tags", ())), d.get("meta", {})))
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def dataset_hash(cases: Sequence[Case]) -> str:
    """Identifies the exact eval set (ids, inputs, expectations, tags). Two runs are only comparable if this matches."""
    blob = json.dumps([[c.id, c.input, c.expected, list(c.tags)] for c in cases], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# -- outputs and scores ---------------------------------------------------------------------------


@dataclass
class Output:
    text: str = ""
    meta: dict = field(default_factory=dict)  # e.g. {"retrieved": [...], "tokens": 123, "cost_usd": 0.001}


@dataclass(frozen=True)
class Score:
    value: float  # 0..1
    passed: bool
    detail: str = ""


class Scorer(Protocol):
    name: str

    def __call__(self, case: Case, output: Output) -> Score: ...


class FnScorer:
    def __init__(self, name: str, fn: Callable[[Case, Output], Score]):
        self.name, self._fn = name, fn

    def __call__(self, case: Case, output: Output) -> Score:
        return self._fn(case, output)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def exact_match() -> Scorer:
    return FnScorer("exact", lambda c, o: Score(float(_norm(o.text) == _norm(c.expected)), _norm(o.text) == _norm(c.expected)))


def contains() -> Scorer:
    """Pass if every expected substring (a string or a list of strings) occurs in the output."""
    def fn(c: Case, o: Output) -> Score:
        wanted = [c.expected] if isinstance(c.expected, str) else list(c.expected or [])
        missing = [w for w in wanted if _norm(w) not in _norm(o.text)]
        return Score(1.0 - len(missing) / max(len(wanted), 1), not missing, f"missing: {missing}" if missing else "")
    return FnScorer("contains", fn)


def regex(pattern: str, name: str = "regex") -> Scorer:
    rx = re.compile(pattern, re.S)
    return FnScorer(name, lambda c, o: Score(float(bool(rx.search(o.text))), bool(rx.search(o.text))))


def json_valid(required: Sequence[str] = (), name: str = "json_valid") -> Scorer:
    """Output parses as a JSON object containing the required keys."""
    def fn(c: Case, o: Output) -> Score:
        try:
            obj = json.loads(o.text)
        except json.JSONDecodeError as e:
            return Score(0.0, False, f"invalid JSON: {e.msg}")
        if not isinstance(obj, dict):
            return Score(0.0, False, "not a JSON object")
        missing = [k for k in required if k not in obj]
        return Score(1.0 - len(missing) / max(len(required), 1) if required else 1.0, not missing, f"missing keys: {missing}" if missing else "")
    return FnScorer(name, fn)


def hit_at_k(k: int = 5, key: str = "retrieved") -> Scorer:
    """Retrieval check: any expected id appears in the first k entries of output.meta[key]."""
    def fn(c: Case, o: Output) -> Score:
        gold = {c.expected} if isinstance(c.expected, str) else set(c.expected or [])
        hit = bool(gold & set(o.meta.get(key, [])[:k]))
        return Score(float(hit), hit)
    return FnScorer(f"hit@{k}", fn)


def grounded(key: str = "cited", allowed_key: str = "retrieved") -> Scorer:
    """Every id the system cited must be one it actually retrieved: catches invented citations."""
    def fn(c: Case, o: Output) -> Score:
        cited, allowed = list(o.meta.get(key, [])), set(o.meta.get(allowed_key, []))
        bad = [x for x in cited if x not in allowed]
        return Score(float(not bad), not bad, f"ungrounded: {bad}" if bad else "")
    return FnScorer("grounded", fn)


# -- running --------------------------------------------------------------------------------------


class System(Protocol):
    id: str

    def __call__(self, case: Case) -> Output: ...


@dataclass
class Record:
    case_id: str
    repeat: int
    text: str
    meta: dict
    error: str | None
    latency_ms: float
    scores: dict[str, dict]


@dataclass
class Run:
    system_id: str
    dataset_hash: str
    scorers: list[str]
    repeats: int
    tags: dict[str, list[str]]  # case id -> tags
    records: list[Record]
    created: float = field(default_factory=time.time)
    info: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_json(text: str) -> Run:
        d = json.loads(text)
        d["records"] = [Record(**r) for r in d["records"]]
        return Run(**d)

    def case_ids(self) -> list[str]:
        return list(dict.fromkeys(r.case_id for r in self.records))

    def _by_case(self) -> dict[str, list[Record]]:
        out: dict[str, list[Record]] = {}
        for r in self.records:
            out.setdefault(r.case_id, []).append(r)
        return out

    def values(self, scorer: str, ids: Sequence[str] | None = None) -> np.ndarray:
        """Per-case score, averaged over repeats; an errored attempt counts as 0."""
        by = self._by_case()
        return np.array([np.mean([r.scores[scorer]["value"] if r.scores.get(scorer) else 0.0 for r in by[i]]) for i in (ids or self.case_ids())])

    def passed(self, scorer: str, ids: Sequence[str] | None = None) -> np.ndarray:
        """Per-case pass/fail: majority vote over repeats (ties count as a fail); an errored attempt counts as a fail."""
        by = self._by_case()
        out = []
        for i in ids or self.case_ids():
            votes = [bool(r.scores.get(scorer) and r.scores[scorer]["passed"]) for r in by[i]]
            out.append(sum(votes) * 2 > len(votes))
        return np.array(out, dtype=bool)

    def flaky(self, scorer: str) -> float:
        """Share of cases whose repeats disagree on pass/fail. Undefined (0) with a single repeat."""
        if self.repeats < 2:
            return 0.0
        by = self._by_case()
        mixed = 0
        for recs in by.values():
            votes = {bool(r.scores.get(scorer) and r.scores[scorer]["passed"]) for r in recs}
            mixed += len(votes) > 1
        return mixed / max(len(by), 1)

    def error_rate(self) -> float:
        return sum(r.error is not None for r in self.records) / max(len(self.records), 1)

    def latency_p95(self) -> float:
        ms = [r.latency_ms for r in self.records if r.error is None]
        return float(np.percentile(ms, 95)) if ms else 0.0


def run(system: System, cases: Sequence[Case], scorers: Sequence[Scorer], repeats: int = 1, timeout: float | None = None, workers: int = 1,
        info: dict | None = None) -> Run:
    """Run every case `repeats` times. A system exception or a timeout becomes an errored record (scored as a failure), never a crash."""
    jobs = [(c, r) for c in cases for r in range(repeats)]

    def one(job: tuple[Case, int]) -> Record:
        case, rep = job
        t0 = time.perf_counter()
        try:
            out = system(case)
            err = None
        except Exception as exc:  # noqa: BLE001 - a crashing system is a result, not a harness failure
            out, err = Output(), f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000
        scores = {}
        if err is None:
            for s in scorers:
                try:
                    sc = s(case, out)
                    scores[s.name] = {"value": sc.value, "passed": sc.passed, "detail": sc.detail}
                except Exception as exc:  # noqa: BLE001
                    scores[s.name] = {"value": 0.0, "passed": False, "detail": f"scorer error: {exc}"}
        return Record(case.id, rep, out.text, out.meta, err, ms, scores)

    def guarded(job: tuple[Case, int]) -> Record:
        if timeout is None:
            return one(job)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(one, job)
            try:
                return fut.result(timeout)
            except concurrent.futures.TimeoutError:
                return Record(job[0].id, job[1], "", {}, f"timeout after {timeout}s", timeout * 1000, {})

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(guarded, jobs))
    else:
        records = [guarded(j) for j in jobs]
    return Run(getattr(system, "id", type(system).__name__), dataset_hash(cases), [s.name for s in scorers], repeats,
               {c.id: list(c.tags) for c in cases}, records, info=info or {})
