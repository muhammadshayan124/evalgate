# evalgate

Regression evaluation for LLM applications, built around one idea: **"the difference is not statistically significant" is not
evidence that nothing got worse.** With 100 test cases, a 10-point drop is routinely "not significant". A CI gate that waves
changes through on that basis is not guarding anything.

`evalgate` runs a system over an eval set, compares a candidate to a baseline on the *same cases*, and blocks a merge unless
it can show, with stated confidence, that the candidate is **non-inferior** within a margin you chose. It also reports how large
a regression the eval set could have caught at all, flags flaky cases, runs meaning-preserving robustness tests, and measures
whether an LLM judge can be trusted before you let it score anything.

It knows nothing about what is being evaluated: a system is any callable from a case to an output (a prompt, a RAG pipeline, an
HTTP endpoint, a local model).

## The gate, on a worked example

A tiny deterministic retrieval example ships in `examples/rag` (100 cases; the corpus is synthetic and exists only to give the harness
cases with known right answers). Three versions of the retriever:

| version | what changed | what the gate should do |
|---|---|---|
| `v1` | baseline | n/a |
| `v1_tuned` | BM25 `k1` 1.2 -> 1.4 | pass (harmless) |
| `v2_bug` | the stemmer stops handling `-ing` words | **fail** (real regression on about 10% of questions) |

```console
$ evalgate run --system examples.rag.systems:v1 --cases examples/rag/cases.jsonl --scorers "hit@3,grounded" --out runs/v1.json
$ evalgate run --system examples.rag.systems:v2_bug ... --out runs/v2_bug.json
$ evalgate gate runs/v1.json runs/v2_bug.json --policy examples/rag/gate.yml          # exit code 1
```

```
## evalgate: FAIL
| non-inferiority: hit@3 | **FAIL** | baseline 0.950 -> candidate 0.870 (diff -0.080); one-sided 95% lower bound -0.150 must be at least -0.020;
                                      pass/fail flips: 12 regressions, 4 improvements (McNemar p=0.077) |
| critical cases (12)    | **FAIL** | 2 critical cases regressed: ['q-v000', 'q-v001'] |
_Sensitivity: with 100 cases ... this eval set detects a true pass-rate drop of about 7.9% with 80% power. A smaller regression can pass._
```

Look at the McNemar line: 12 regressions against 4 improvements gives **p = 0.077, "not significant"**. A significance-test gate would pass
this candidate. The non-inferiority gate fails it, correctly, because the data cannot exclude a drop larger than the 2-point margin
(the one-sided lower bound is -15 points). That is the whole point of the design.

## How the decision is made

| check | what it asserts |
|---|---|
| **same eval set** | the two runs share an eval-set hash (ids, inputs, expectations, tags); otherwise they are not comparable and the gate refuses |
| **non-inferiority** (per metric) | the one-sided (1 - alpha) lower bound of `candidate - baseline` on per-case scores, from a paired bootstrap over cases, is at least `-margin` |
| **floor** | the candidate's absolute pass rate clears a minimum (Wilson interval shown) |
| **critical cases** | no case tagged `critical` may go from pass to fail, however good the average is |
| **error rate / latency p95** | crashes and timeouts are failures, not exceptions; latency may grow by a bounded percentage |
| **flakiness** | with `--repeats N`, the share of cases whose repeats disagree stays under a limit; per-case outcome is the majority vote |
| **sensitivity note** | the smallest drop this eval set detects at 80% power, and a warning when your margin is finer than the set can resolve |

`evalgate power --n 100 --delta 0.03` answers the planning question directly: detecting a 3-point drop needs about **687 cases**;
1,000 cases detect about 2.5 points; 100 cases detect about 8.

### The error rates are tested, not asserted

`tests/test_evalgate.py::test_gate_error_rates_match_the_stated_confidence` simulates thousands of version pairs with a known true drop
and checks the gate behaves as the statistics say (400 cases, 10% discordant, margin 0.05): a regression *exactly at the margin* is
accepted only about 5% of the time (alpha); a regression of twice the margin is essentially never accepted (< 1%); an unchanged
candidate is accepted about 94% of the time. That last number is the test's **power**, not a false-alarm rate: even a good candidate is
blocked by chance a few percent of the time, and the way to reduce that is more cases, not a looser alpha.

## The rest of the toolkit

- **Scorers**: exact, contains, regex, JSON validity with required keys, retrieval `hit@k`, and `grounded` (every cited id was actually
  retrieved, which catches invented citations). Any function `(case, output) -> Score` works.
- **Robustness** (`evalgate robustness`): metamorphic tests that change the input without changing its meaning (typos, casing, whitespace,
  an irrelevant appended sentence) and report the pass-rate change with an interval and the cases that newly fail. All perturbations are
  benign; nothing here generates adversarial or harmful text.
- **Judge calibration** (`evalgate calibrate`, `judge.position_bias`): an LLM judge is a classifier with an accuracy. Measure agreement
  and Cohen's kappa against trusted labels, the **false-pass rate** (the dangerous error for a quality gate), and whether the verdict
  depends on the order the two answers are shown in.
- **Reproducible runs**: a run is a JSON file (system id, eval-set hash, every output, score and latency), so a baseline can be cached
  as a CI artifact and compared later.

## Use it in CI

```yaml
- uses: actions/checkout@v4
- run: pip install evalgate
- run: evalgate run --system myapp.eval:candidate --cases evals/cases.jsonl --scorers "contains,grounded" --out candidate.json
- run: evalgate gate baseline.json candidate.json --policy evals/gate.yml --markdown "$GITHUB_STEP_SUMMARY"
```

The gate exits 1 on failure and appends the Markdown report (with the regressed case ids and their reasons) to the job summary.

## Limitations

- **The eval set bounds everything.** The gate is only as meaningful as the cases; it measures regression on *those* cases, not on
  production traffic. The sensitivity note exists so that is never forgotten.
- **Paired bootstrap on independent cases.** If your cases are clustered (many questions about one document), the real uncertainty is
  larger than the bootstrap reports; group such cases or lower your confidence in a green gate.
- **Binary pass/fail is the main currency.** Continuous scores are supported and averaged, but the McNemar test and the critical-case check
  use the pass/fail flag.
- **The example corpus is synthetic.** It shows the mechanics and is deliberately easy; it says nothing about real retrieval quality.
- **LLM judges are only as good as their calibration.** `calibrate` tells you how good; it cannot make a weak judge strong.

## Develop

```bash
pip install -e ".[dev]"
pytest          # 26 tests, including the gate-calibration simulations
```

MIT licence.
