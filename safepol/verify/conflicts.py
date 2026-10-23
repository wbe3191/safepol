"""Exact rule-conflict analysis (Al-Shaer taxonomy) with an SMT encoding.

Two encodings are provided and cross-validated:
  * ``fast``  : product-form reasoning on subject/peer sets and port-class masks
                (sound and complete for rules whose predicate is a product of sets);
  * ``smt``   : Z3 encoding with src,dst as integers and (proto,port) as a 16-bit
                bit-vector over *raw* ports; implication phi_j => phi_i is decided as
                UNSAT(phi_j & ~phi_i).  This is the reference semantics of Sec. 1.4.

Under Cilium semantics deny has precedence over allow, so for an (allow, deny) pair the
deny rule is the higher-priority rule irrespective of numeric priority.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from safepol.model.policy_model import ALLOW, DENY, Policy, Rule, Topology

SHADOW, REDUND, GENERAL, CORREL = "shadowing", "redundancy", "generalization", "correlation"


@dataclass
class Conflict:
    hi: int      # rid of higher-priority rule
    lo: int      # rid of lower-priority rule
    kind: str


def _ordered(a: Rule, b: Rule) -> Tuple[Rule, Rule]:
    """Return (higher, lower) priority rule."""
    if a.verdict != b.verdict:
        return (a, b) if a.verdict == DENY else (b, a)
    if a.priority != b.priority:
        return (a, b) if a.priority > b.priority else (b, a)
    return (a, b) if a.rid < b.rid else (b, a)


# -------------------------------------------------------------------------------------
# fast product-form analysis
# -------------------------------------------------------------------------------------
def _implies(j: Rule, i: Rule) -> bool:
    return j.subject <= i.subject and j.peer <= i.peer and (j.ports & ~i.ports) == 0


def _sat_and(i: Rule, j: Rule) -> bool:
    return bool(i.subject & j.subject) and bool(i.peer & j.peer) and bool(i.ports & j.ports)


def classify_pair_fast(a: Rule, b: Rule) -> Optional[Conflict]:
    if a.direction != b.direction or not (a.subject & b.subject):
        return None
    hi, lo = _ordered(a, b)
    if not _sat_and(hi, lo):
        return None
    lo_in_hi = _implies(lo, hi)
    hi_in_lo = _implies(hi, lo)
    if lo_in_hi:
        return Conflict(hi.rid, lo.rid, SHADOW if hi.verdict != lo.verdict else REDUND)
    if hi_in_lo and hi.verdict != lo.verdict:
        return Conflict(hi.rid, lo.rid, GENERAL)
    if hi.verdict != lo.verdict:
        return Conflict(hi.rid, lo.rid, CORREL)
    return None


def _candidate_pairs(rules: List[Rule], only_subjects=None):
    """Pairs that can conflict at all: same direction and intersecting subject selectors
    (Sec. 2.3 performance engineering (a)).  Indexed by (direction, subject workload) so the
    number of pairs examined is sum over workloads of (#rules selecting it)^2, not |R|^2."""
    idx: Dict[Tuple[str, int], List[int]] = {}
    for i, r in enumerate(rules):
        for w in r.subject:
            if only_subjects is None or w in only_subjects:
                idx.setdefault((r.direction, w), []).append(i)
    seen = set()
    for bucket in idx.values():
        for a in range(len(bucket)):
            for b in range(a + 1, len(bucket)):
                key = (bucket[a], bucket[b])
                if key not in seen:
                    seen.add(key)
                    yield key


def detect_conflicts(P: Policy, method: str = "fast", only_pairs_with=None) -> List[Conflict]:
    """Conflicts among the rules of P.  ``only_pairs_with``: optional set of workloads; when given,
    only pairs of rules selecting at least one of them are examined (localised check)."""
    rules = P.rules
    out: List[Conflict] = []
    if method == "fast":
        for i, j in _candidate_pairs(rules, only_pairs_with):
            c = classify_pair_fast(rules[i], rules[j])
            if c:
                out.append(c)
        return out
    smt = SMTConflictAnalyzer(P.topo)
    for i, j in _candidate_pairs(rules, only_pairs_with):
        c = smt.classify_pair(rules[i], rules[j])
        if c:
            out.append(c)
    return out


def harmful(conflicts: List[Conflict]) -> List[Conflict]:
    """Conf(P): shadowing and correlation anomalies (generalization/redundancy are benign)."""
    return [c for c in conflicts if c.kind in (SHADOW, CORREL)]


# -------------------------------------------------------------------------------------
# SMT encoding
# -------------------------------------------------------------------------------------
class SMTConflictAnalyzer:
    def __init__(self, topo: Topology):
        import z3  # local import keeps the fast path dependency-free
        self.z3 = z3
        self.topo = topo
        self.src = z3.Int("src")
        self.dst = z3.Int("dst")
        self.proto = z3.BitVec("proto", 1)
        self.port = z3.BitVec("port", 16)
        self._cache: Dict[Tuple[int, int], bool] = {}
        self.calls = 0
        self.solver_time = 0.0

    def predicate(self, r: Rule):
        z3 = self.z3
        subj = z3.Or([ (self.src if r.direction == "out" else self.dst) == i for i in r.subject ])
        peer_var = self.dst if r.direction == "out" else self.src
        peer = z3.Or([peer_var == i for i in r.peer]) if r.peer else z3.BoolVal(False)
        port_terms = []
        for (proto, lo, hi) in self.topo.ps.intervals(r.ports):
            port_terms.append(z3.And(self.proto == proto,
                                     z3.UGE(self.port, lo), z3.ULE(self.port, hi)))
        ports = z3.Or(port_terms) if port_terms else z3.BoolVal(False)
        return z3.And(subj, peer, ports)

    def _sat(self, *fs) -> bool:
        z3 = self.z3
        s = z3.Solver()
        s.add(*fs)
        t0 = time.perf_counter()
        res = s.check()
        self.solver_time += time.perf_counter() - t0
        self.calls += 1
        return res == z3.sat

    def implies(self, j: Rule, i: Rule) -> bool:
        key = (j.rid, i.rid)
        if key not in self._cache:
            self._cache[key] = not self._sat(self.predicate(j), self.z3.Not(self.predicate(i)))
        return self._cache[key]

    def classify_pair(self, a: Rule, b: Rule) -> Optional[Conflict]:
        if a.direction != b.direction or not (a.subject & b.subject):
            return None
        hi, lo = _ordered(a, b)
        if not self._sat(self.predicate(hi), self.predicate(lo)):
            return None
        if self.implies(lo, hi):
            return Conflict(hi.rid, lo.rid, SHADOW if hi.verdict != lo.verdict else REDUND)
        if self.implies(hi, lo) and hi.verdict != lo.verdict:
            return Conflict(hi.rid, lo.rid, GENERAL)
        if hi.verdict != lo.verdict:
            return Conflict(hi.rid, lo.rid, CORREL)
        return None


def cross_validate(P: Policy) -> Tuple[bool, int, int]:
    """Check that fast and SMT classifications agree. Returns (agree, n_fast, n_smt)."""
    f = {(c.hi, c.lo, c.kind) for c in detect_conflicts(P, "fast")}
    s = {(c.hi, c.lo, c.kind) for c in detect_conflicts(P, "smt")}
    return f == s, len(f), len(s)
