"""A small, fully deterministic retrieval example used by the docs and the tests.

The corpus is synthetic (120 short support-policy passages, each about one distinctive topic word) and the questions
paraphrase the topic word into a different inflection than the passage uses ("refund" asked, "refunds" in the passage).
That makes the stemmer the thing that matters, so we can introduce known regressions and check that the gate catches them:

  v1        BM25 with a stemmer                       the baseline
  v1_tuned  the same with a slightly different k1      a harmless change (the gate should pass it)
  v2_bug    the stemmer forgets '-ing' words           a real regression on about 10% of questions (the gate should fail it)

Nothing here is a claim about real retrieval quality; it exists to exercise the harness on cases whose right answer is known.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from evalgate.core import Case, Output

NOUNS = ("refund invoice warranty shipment password account receipt coupon subscription cancellation delivery exchange voucher balance deposit "
         "statement transfer deadline appointment reservation booking ticket license permit certificate registration enrollment membership "
         "complaint inquiry feedback survey review rating comment message notification reminder alert update upgrade backup archive "
         "export import download upload attachment document template contract agreement amendment extension renewal discount promotion "
         "rebate surcharge tariff quota allowance threshold limit penalty fine deduction claim policy premium benefit coverage exclusion "
         "deductible liability dispute appeal ruling hearing inspection audit report ledger journal budget forecast estimate quotation").split()
VERBS = "reimburse reschedule redeem reactivate relocate reissue renegotiate reconcile resubmit reorder recharge reassign".split()  # passages use '-ing'


def _gerund(v: str) -> str:
    return (v[:-1] if v.endswith("e") else v) + "ing"


def corpus() -> dict[str, str]:
    docs = {}
    for i, n in enumerate(NOUNS):
        docs[f"d{i:03d}"] = f"Our policy on {n}s explains who may use them, how long they stay valid and where to find the details in your account area."
    for j, v in enumerate(VERBS):
        docs[f"v{j:03d}"] = f"The rules for {_gerund(v)} items describe the steps, the fees involved and the time limits that apply to customers."
    return docs


def cases() -> list[Case]:
    out = []
    for i, n in enumerate(NOUNS):
        out.append(Case(f"q-n{i:03d}", f"What should I know about a {n}?", f"d{i:03d}", tags=("critical",) if i < 10 else ()))
    for j, v in enumerate(VERBS):
        out.append(Case(f"q-v{j:03d}", f"How do I {v} something?", f"v{j:03d}", tags=("critical",) if j < 2 else ()))
    return out


STOP = frozenset("a an and are as at be by do how i in is it of on or the to what where who which should know about something".split())


def make_stemmer(strip_ing: bool = True):
    def stem(w: str) -> str:
        if strip_ing and w.endswith("ing") and len(w) > 5:
            w = w[:-3]
            return w[:-1] if w.endswith("e") else w
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        return w[:-1] if w.endswith("e") and len(w) > 4 else w
    return stem


class BM25System:
    def __init__(self, id: str, k1: float = 1.2, b: float = 0.75, strip_ing: bool = True):
        self.id, self.k1, self.b = id, k1, b
        self.stem = make_stemmer(strip_ing)
        self.docs = {k: self._tok(v) for k, v in corpus().items()}
        self.avg = sum(len(d) for d in self.docs.values()) / len(self.docs)
        df = Counter(t for d in self.docs.values() for t in set(d))
        self.idf = {t: math.log(1 + (len(self.docs) - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _tok(self, text: str) -> list[str]:
        return [self.stem(t) for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP]

    def __call__(self, case: Case) -> Output:
        q = set(self._tok(str(case.input)))
        scored = []
        for k, d in self.docs.items():
            tf = Counter(d)
            s = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg)) for t in q if t in tf)
            scored.append((s, k))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [k for s, k in scored[:5] if s > 0]
        return Output(top[0] if top else "", {"retrieved": top, "cited": top[:1]})


v1 = BM25System("bm25-v1")
v1_tuned = BM25System("bm25-v1-tuned", k1=1.4)
v2_bug = BM25System("bm25-v2-ing-bug", strip_ing=False)
