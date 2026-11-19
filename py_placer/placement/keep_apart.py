"""Keep-apart rules: distances that move with the parts.

A keep-apart rule names ANCHOR parts and the parts they keep away from, and a
gap: a DC-DC regulator's parts kept 5 mm from every connector whose cable
leaves the box, a switching coil kept 1.27 mm from every part outside its own
regulator group. An intent keep-out is a fixed area; these follow the parts,
so they hold while the anneal moves both sides of the pair.

The file (`place_simple.py --keep-apart FILE`, `check_keep_apart.py --rules
FILE`):

    {"schema": 1, "rules": [
      {"name": "dcdc_vs_cable_connectors", "parts": ["CN2", "CN5"],
       "from": ["U16", "L2", "C61"], "gap_mm": 5.0,
       "same_face": "courtyard", "opposite_face": "courtyard", "why": "..."},
      {"name": "switching_coil_clearance", "parts": ["L2"], "from": "strangers",
       "gap_mm": 1.27, "same_face": "courtyard", "opposite_face": "net_pads"}]}

`parts` are the anchors; `from` a list of refs, or "strangers": every part
outside the groups the anchor is a member of (the groups file as place_simple
places it, after the claim fold). A pair is measured on the face relation of
the two parts (`same_face` when they stand on one board face, else
`opposite_face`):

    courtyard   the gap between the two courtyards (the grade's rect:
                courtyard, else pad box)
    net_pads    the gap from the anchor's courtyard to the other part's
                net-carrying pad copper (a body, or a net-less mechanical
                pad, may stand there)
    none        not measured

Every gap is the SIGNED Euclidean gap between rectangles: the distance
between them when apart, minus the shallower penetration when they overlap,
so a part under another is pushed out the short way. A pair breaking its
rule costs `weight x shortfall^2`, the shortfall `gap - distance` capped at
`gap + DEEP_MM` (a part deeper inside pays as one DEEP_MM inside); it is
BROKEN while the distance is more than TOL_MM short of the gap.

One implementation measures for both readers: the anneal term
(`simple.Model`, priced incrementally) and the grader
(`py_tools/check_keep_apart.py`), so a board is graded as it was annealed.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 1
MODES = ('courtyard', 'net_pads', 'none')
#: a pair is broken while it is more than this many mm short of its gap: a
#: quadratic price leaves a vanishing shortfall where a wire pulls against it
TOL_MM = 0.01
#: mm of penetration past which a deeper part pays no more. The shallower
#: penetration is at most half the smaller part, so this only bounds the
#: price of a part deep under a very large one; a flat price inside would
#: give the anneal no way out (Orin at 5 mm: a 17 mm coil kept a choke 7.7 mm
#: under it, where every nearby pose paid the same)
DEEP_MM = 25.0

_TOP_KEYS = {'schema', 'rules', 'why'}
_RULE_KEYS = {'name', 'parts', 'from', 'gap_mm', 'same_face', 'opposite_face', 'why'}
_RULE_REQUIRED = _RULE_KEYS - {'why'}


class RulesError(ValueError):
    """A keep-apart file that cannot be used, with the reason."""


def _refs(v, where):
    if (not isinstance(v, list) or not v
            or not all(isinstance(r, str) and r for r in v)):
        raise RulesError(f"{where} must be a non-empty list of refs")
    if len(set(v)) != len(v):
        dup = sorted({r for r in v if v.count(r) > 1})
        raise RulesError(f"{where} names {', '.join(dup)} more than once")
    return list(v)


def validate(doc, board_refs: Optional[Iterable[str]] = None) -> List[Dict]:
    """The rules of a parsed keep-apart file, checked strictly: unknown keys,
    a missing key, a bad mode or gap, and (with `board_refs`) a ref the board
    does not have are refused with a RulesError naming the rule and key."""
    if not isinstance(doc, dict):
        raise RulesError("the file must hold a JSON object")
    extra = sorted(set(doc) - _TOP_KEYS)
    if extra:
        raise RulesError(f"unknown top-level key(s) {', '.join(extra)}; "
                         f"allowed: {', '.join(sorted(_TOP_KEYS))}")
    if doc.get('schema') != SCHEMA:
        raise RulesError(f"'schema' must be {SCHEMA}, got {doc.get('schema')!r}")
    rules = doc.get('rules')
    if not isinstance(rules, list) or not rules:
        raise RulesError("'rules' must be a non-empty list")
    known = set(board_refs) if board_refs is not None else None
    out, names = [], set()
    for k, r in enumerate(rules):
        where = f"rules[{k}]"
        if not isinstance(r, dict):
            raise RulesError(f"{where} must be an object")
        extra = sorted(set(r) - _RULE_KEYS)
        if extra:
            raise RulesError(f"{where}: unknown key(s) {', '.join(extra)}; "
                             f"allowed: {', '.join(sorted(_RULE_KEYS))}")
        miss = sorted(_RULE_REQUIRED - set(r))
        if miss:
            raise RulesError(f"{where}: missing key(s) {', '.join(miss)}")
        name = r['name']
        if not isinstance(name, str) or not name:
            raise RulesError(f"{where}: 'name' must be a non-empty string")
        if name in names:
            raise RulesError(f"{where}: the name {name!r} is used twice")
        names.add(name)
        where = f"rule {name!r}"
        parts = _refs(r['parts'], f"{where}: 'parts'")
        frm = r['from']
        if frm != 'strangers':
            if isinstance(frm, str):
                raise RulesError(f"{where}: 'from' must be a list of refs or \"strangers\", "
                                 f"got {frm!r}")
            frm = _refs(frm, f"{where}: 'from'")
        gap = r['gap_mm']
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) \
                or not math.isfinite(gap) or gap <= 0:
            raise RulesError(f"{where}: 'gap_mm' must be a number above 0, got {gap!r}")
        for key in ('same_face', 'opposite_face'):
            if r[key] not in MODES:
                raise RulesError(f"{where}: {key!r} must be one of {', '.join(MODES)}, "
                                 f"got {r[key]!r}")
        if r['same_face'] == 'none' and r['opposite_face'] == 'none':
            raise RulesError(f"{where}: both faces are 'none', so it measures nothing")
        if 'why' in r and not isinstance(r['why'], str):
            raise RulesError(f"{where}: 'why' must be a string")
        if known is not None:
            bad = sorted((set(parts) | (set(frm) if frm != 'strangers' else set())) - known)
            if bad:
                raise RulesError(f"{where}: the board has no {', '.join(bad)}")
        out.append({'name': name, 'parts': parts, 'from': frm, 'gap_mm': float(gap),
                    'same_face': r['same_face'], 'opposite_face': r['opposite_face']})
    return out


def load(path: str, board_refs: Optional[Iterable[str]] = None) -> List[Dict]:
    """`validate` of the JSON file at `path`."""
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise RulesError(f"cannot read {path}: {e}") from None
    return validate(doc, board_refs)


# -- measurement --------------------------------------------------------------
def signed_gap(a, b) -> float:
    """The signed Euclidean gap between rects `a` and `b` (x0, y0, x1, y1):
    their distance when apart, minus the shallower penetration when they
    overlap (0 when they touch)."""
    dx = max(b[0] - a[2], a[0] - b[2])
    dy = max(b[1] - a[3], a[1] - b[3])
    if dx > 0.0 and dy > 0.0:
        return math.hypot(dx, dy)
    return max(dx, dy)


def signed_gaps(a, B: np.ndarray) -> np.ndarray:
    """`signed_gap` of rect `a` against each row of `B` (k x 4)."""
    dx = np.maximum(B[:, 0] - a[2], a[0] - B[:, 2])
    dy = np.maximum(B[:, 1] - a[3], a[1] - B[:, 3])
    return np.where((dx > 0.0) & (dy > 0.0), np.hypot(dx, dy), np.maximum(dx, dy))


def shortfall_cost(gap: float, d: float, weight: float) -> float:
    """`weight x shortfall^2` of a pair at distance `d` from a `gap` rule."""
    s = gap - d
    if s <= 0.0:
        return 0.0
    s = min(s, gap + DEEP_MM)
    return weight * s * s


class Geometry:
    """What a pair is measured on, read live from a placement state's parts:
    each part's face, courtyard rect (`grade_rect`: courtyard, else pad box)
    and net-carrying pad copper (`pads(ref)`, k x 4, the caller's)."""

    def __init__(self, parts, pads):
        self.parts, self.pads = parts, pads

    def side(self, r):
        return self.parts[r].side

    def crt(self, r):
        return tuple(self.parts[r].grade_rect()[:4])

    def presence(self, r):
        """Everything of `r` a rule can measure: its courtyard and its net
        pads' copper, as one box."""
        c = self.crt(r)
        P = self.pads(r)
        if not len(P):
            return c
        return (min(c[0], float(P[:, 0].min())), min(c[1], float(P[:, 1].min())),
                max(c[2], float(P[:, 2].max())), max(c[3], float(P[:, 3].max())))


class KeepApart:
    """The rules resolved against a board: per rule its anchors and, per
    anchor, the parts it keeps away from (an explicit set, or every part but
    its own groups' members for "strangers")."""

    def __init__(self, rules: Sequence[Dict], groups: Sequence[Tuple[str, Sequence[str]]],
                 refs: Iterable[str]):
        refs = set(refs)
        self.refs = refs
        member: Dict[str, set] = {}
        for _name, members in groups:
            for r in members:
                member.setdefault(r, set()).update(members)
        self.rules = []
        self.anchor_rules: Dict[str, List[int]] = {}
        self.dropped: List[str] = []
        explicit: set = set()
        any_strangers = False
        for i, r in enumerate(rules):
            anchors = tuple(a for a in r['parts'] if a in refs)
            self.dropped += [a for a in r['parts'] if a not in refs]
            if r['from'] == 'strangers':
                frm = None
                own = {a: frozenset(member.get(a, ()) | {a}) for a in anchors}
                any_strangers = True
            else:
                frm = frozenset(b for b in r['from'] if b in refs)
                self.dropped += [b for b in r['from'] if b not in refs]
                own = {a: frozenset((a,)) for a in anchors}
                explicit |= frm
            self.rules.append((r['name'], r['gap_mm'], r['same_face'], r['opposite_face'],
                               anchors, frm, own))
            for a in anchors:
                self.anchor_rules.setdefault(a, []).append(i)
        self.anchors = frozenset(self.anchor_rules)
        #: the parts that can be the far end of a pair (None: any part)
        self.others = None if any_strangers else frozenset(explicit)
        self.reach = {a: max(self.rules[i][1] for i in ix)
                      for a, ix in self.anchor_rules.items()}

    def other(self, r) -> bool:
        return self.others is None or r in self.others

    def wants(self, i, a, b) -> bool:
        _n, _g, _s, _o, _anc, frm, own = self.rules[i]
        if b == a:
            return False
        return b not in own[a] if frm is None else b in frm

    def distance(self, i, a, b, geom: Geometry) -> Optional[float]:
        """Rule `i`'s distance from anchor `a` to `b` (None: not measured on
        their face relation, or `b` has no net pad to measure)."""
        _n, _g, same, opp, _anc, _frm, _own = self.rules[i]
        mode = same if geom.side(a) == geom.side(b) else opp
        if mode == 'courtyard':
            return signed_gap(geom.crt(a), geom.crt(b))
        if mode == 'net_pads':
            P = geom.pads(b)
            if not len(P):
                return None
            return float(signed_gaps(geom.crt(a), P).min())
        return None

    def pair(self, a, b, geom: Geometry, weight: float) -> List[Tuple[int, float, float]]:
        """`[(rule, distance, cost)]` of anchor `a` against `b`, each rule
        that wants the pair and measures it."""
        out = []
        for i in self.anchor_rules.get(a, ()):
            if not self.wants(i, a, b):
                continue
            d = self.distance(i, a, b, geom)
            if d is None:
                continue
            out.append((i, d, shortfall_cost(self.rules[i][1], d, weight)))
        return out

    def near(self, a, box, geom: Geometry) -> bool:
        """Whether `box` (a part's `presence`) can be within reach of anchor
        `a`: inside its courtyard grown by its widest gap."""
        c, g = geom.crt(a), self.reach[a]
        return not (box[2] < c[0] - g or box[0] > c[2] + g
                    or box[3] < c[1] - g or box[1] > c[3] + g)

    def scan(self, geom: Geometry, weight: float):
        """Every pair from scratch: `(cost, rows)`, rows `(rule, anchor,
        other, distance, cost)` for each measured pair inside reach, in
        anchor then other order."""
        names = sorted(self.refs if self.others is None else self.others)
        boxes = {b: geom.presence(b) for b in names}
        total, rows = 0.0, []
        for a in sorted(self.anchors):
            for b in names:
                if b == a or not self.near(a, boxes[b], geom):
                    continue
                for i, d, v in self.pair(a, b, geom, weight):
                    total += v
                    rows.append((i, a, b, d, v))
        return total, rows

    def broken(self, rows) -> List[List]:
        """The rows of `scan` that break their rule: `[rule, anchor, other,
        distance_mm]`, the distance rounded to the micron."""
        return [[self.rules[i][0], a, b, round(d, 3)] for i, a, b, d, _v in rows
                if d < self.rules[i][1] - TOL_MM]
