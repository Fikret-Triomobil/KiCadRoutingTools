#!/usr/bin/env python3
"""Grade a placed board against keep-apart rules (`place_simple --keep-apart`).

Each rule keeps its anchor parts a gap from the parts it names, or from every
part outside the anchor's own groups ("strangers": the groups file as
place_simple places it, after the claim fold -- pass the same `--intent` to
fold it the same way). A pair is measured courtyard to courtyard, or
courtyard to the other part's net pads, by the face relation of the two
parts (`placement/keep_apart.py`, the one measure the anneal's term prices
too), and is BROKEN while it is more than 0.01 mm short of its gap. So the
human reference board and a placer's candidates are graded identically.

    python3 -X utf8 py_tools/check_keep_apart.py board.kicad_pcb --rules keep_apart.json \\
        --groups groups.json [--intent intent.json] [--json report.json]

Each broken pair is printed `rule: anchor - other  d mm (gap g mm)`; a part
under another reads a negative distance (the shallower penetration). Parts
without a drawn courtyard are measured on the estimate place_simple seats
them on (#1300).

Exit codes: 0 no rule broken, 2 unreadable input, 4 a rule broken.
"""
#: #937 registry: which door(s) show this tool, and whether it changes
#: the board. Read by krt_registry.py -- by AST, never imported.
KRT_TOOL = {'scope': ['placement'], 'kind': 'instrument'}

import _path  # noqa: F401  (py_tools -> py_router/py_placer on sys.path)

import argparse
import json
import sys

from kicad_parser import parse_kicad_pcb  # noqa: E402


def grade(pcb, board, rules, groups):
    """`(kap, rows, broken)` of `pcb` (read from `board`) under the
    validated `rules`, "strangers" read from `groups` ([(name, refs)]): the
    rows `KeepApart.scan` measures at weight 1 and the broken ones."""
    import pose_score
    from placement.keep_apart import Geometry, KeepApart
    from placement.simple import conn_rects
    state = pose_score.make_state(pcb, board, estimate_courtyards=True)
    kap = KeepApart(rules, groups, state.parts)
    memo, boxes = {}, {}
    geom = Geometry(state.parts, lambda r: conn_rects(pcb, state.parts, r, memo, boxes))
    _cost, rows = kap.scan(geom, 1.0)
    return kap, rows, kap.broken(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board', help='the placed board to grade')
    ap.add_argument('--rules', required=True, help='the keep-apart rules (JSON)')
    ap.add_argument('--groups', help='the groups file: needed by a "strangers" rule')
    ap.add_argument('--intent', help="the intent place_simple read: its proximity claims "
                    "and templates fold the groups as place_simple folds them")
    ap.add_argument('--json', dest='json_out', help='write the report here')
    a = ap.parse_args(argv)
    from placement.keep_apart import RulesError, load
    try:
        pcb = parse_kicad_pcb(a.board)
    except Exception as ex:  # an unreadable board is an input error, not a finding
        print(f"check_keep_apart: cannot read {a.board}: {ex}", file=sys.stderr)
        return 2
    try:
        rules = load(a.rules, pcb.footprints)
    except RulesError as ex:
        print(f"check_keep_apart: {a.rules}: {ex}", file=sys.stderr)
        return 2
    groups = []
    if a.groups:
        from placement.grouping import GroupsError, claim_rows, fold_groups, load_groups
        try:
            doc = load_groups(a.groups)
            intent_doc, tmpls = {}, []
            if a.intent:
                from placement import floorplan
                with open(a.intent, encoding='utf-8') as f:
                    intent_doc = json.load(f)
                intent = floorplan.load_intent(a.intent)
                tmpls = list(intent.templates) + intent.cover_templates()
        except (GroupsError, OSError, ValueError) as ex:
            print(f"check_keep_apart: {ex}", file=sys.stderr)
            return 2
        rows, _notes = claim_rows(pcb, intent_doc)
        doc = fold_groups(doc, rows, tmpls)['doc']
        groups = [(g['name'], g['refs']) for g in doc['groups']]
    elif any(r['from'] == 'strangers' for r in rules):
        print("check_keep_apart: a \"strangers\" rule needs --groups (whose groups the "
              "anchors' own circuits are)", file=sys.stderr)
        return 2
    kap, rows, broken = grade(pcb, a.board, rules, groups)
    gap = {r[0]: r[1] for r in kap.rules}
    print(f"check_keep_apart: {a.board}: {len(kap.rules)} rule(s), {len(kap.anchors)} "
          f"anchor(s); {len(broken)} pair(s) broken")
    if kap.dropped:
        print(f"  named but not in the placement state: {', '.join(sorted(set(kap.dropped)))}")
    for name, anc, other, d in broken:
        print(f"  {name}: {anc} - {other}  {d:.3f} mm (gap {gap[name]:g} mm)")
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump({'board': a.board, 'rules': a.rules,
                       'broken': broken,
                       'per_rule': {n: sum(1 for b in broken if b[0] == n) for n in gap}},
                      f, indent=1)
    return 4 if broken else 0


if __name__ == '__main__':
    sys.exit(main())
