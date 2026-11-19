"""Keep-apart rules (`--keep-apart FILE`): gaps that move with the parts.

Each rule keeps its anchor parts a gap from the parts it names, or from every
part outside the anchor's own groups ("strangers"), courtyard to courtyard or
courtyard to the other part's net pads by the two parts' face relation, and
a pair short of it pays `weight x shortfall^2` (`placement/keep_apart.py`,
`simple.Model._kap_*`).

Checked here, on esp_prog with three parts turned to the back face:

- the file is validated strictly: an unknown key, a bad mode, a gap of 0 and
  a ref the board lacks are refused, naming the rule;
- the signed gap reads a distance apart, a corner diagonally, and the
  shallower penetration under an overlap;
- a move is priced locally exactly as `total()` prices the board from
  scratch, over a few hundred random moves of single parts and pairs (the
  anneal adds the local changes up, so a drift would steer it), and the
  committed pairs add up to the from-scratch term; both face relations are
  exercised;
- without a file the placer is unchanged: the same seed writes the same
  board and the same final cost (a weight without a file is off), and the
  breakdown carries no `keep_apart`;
- with one, the term and its broken pairs are reported in the log's final
  line, JSON_SUMMARY and --json, and `py_tools/check_keep_apart.py` finds the
  same broken pairs on the written board.
"""

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('py_placer', 'py_router', 'py_tools', 'tests'):
    sys.path.insert(0, os.path.join(ROOT, _p))

import place_simple as ps                                     # noqa: E402
from kicad_parser import parse_kicad_pcb                      # noqa: E402
from placement import floorplan as fp                         # noqa: E402
from placement import keep_apart as ka                        # noqa: E402
from placement.grouping import load_groups                    # noqa: E402
from placement.groups import parse_sources                    # noqa: E402
from placement.simple_anneal import _place                    # noqa: E402

BOARD = os.path.join(ROOT, 'kicad_files', 'esp_prog.kicad_pcb')
GROUPS = {'schema': 1, 'groups': [
    {'name': 'power', 'refs': ['U2', 'C1', 'C3']},
    {'name': 'core', 'refs': ['U1', 'Y1', 'C2', 'C4', 'R3', 'R4']},
    {'name': 'autoreset', 'refs': ['Q1', 'Q2', 'R1', 'R2']}]}
#: turned to the back face, so pairs meet across the board too
BACK = ('C3', 'R1', 'R2')
RULES = {'schema': 1, 'rules': [
    {'name': 'power_vs_connectors', 'parts': ['CON1', 'USB1', 'CON2'],
     'from': ['U2', 'C1', 'C3'], 'gap_mm': 3.0,
     'same_face': 'courtyard', 'opposite_face': 'courtyard', 'why': 'test'},
    {'name': 'coil', 'parts': ['U2'], 'from': 'strangers', 'gap_mm': 1.27,
     'same_face': 'courtyard', 'opposite_face': 'net_pads'}]}
FAILS = []


def ok(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def test_validation():
    refs = set(parse_kicad_pcb(BOARD).footprints)

    def refused(doc, words):
        try:
            ka.validate(doc, refs)
        except ka.RulesError as ex:
            return all(w in str(ex) for w in words)
        return False
    good = json.loads(json.dumps(RULES))
    ok('the test rules pass', len(ka.validate(good, refs)) == 2)
    bad = json.loads(json.dumps(RULES))
    bad['rules'][0]['gap'] = 1
    ok('an unknown rule key is refused', refused(bad, ['power_vs_connectors', 'gap']) or
       refused(bad, ['rules[0]', 'gap']))
    bad = json.loads(json.dumps(RULES))
    bad['extra'] = 1
    ok('an unknown top-level key is refused', refused(bad, ['extra']))
    bad = json.loads(json.dumps(RULES))
    bad['rules'][1]['opposite_face'] = 'pads'
    ok('a bad mode is refused', refused(bad, ["'coil'", 'opposite_face']))
    bad = json.loads(json.dumps(RULES))
    bad['rules'][1]['from'] = 'stranger'
    ok('a misspelt strangers is refused', refused(bad, ["'coil'", 'from']))
    bad = json.loads(json.dumps(RULES))
    bad['rules'][0]['gap_mm'] = 0
    ok('a gap of 0 is refused', refused(bad, ['gap_mm']))
    bad = json.loads(json.dumps(RULES))
    bad['rules'][0]['parts'].append('CN99')
    ok('a ref the board lacks is refused', refused(bad, ['CN99']))
    bad = json.loads(json.dumps(RULES))
    del bad['rules'][0]['same_face']
    ok('a missing key is refused', refused(bad, ['same_face']))


def test_signed_gap():
    a = (0.0, 0.0, 2.0, 2.0)
    ok('apart on one axis', abs(ka.signed_gap(a, (5.0, 0.5, 6.0, 1.0)) - 3.0) < 1e-12)
    ok('apart at a corner, diagonally',
       abs(ka.signed_gap(a, (5.0, 6.0, 7.0, 8.0)) - 5.0) < 1e-12)
    ok('under it: minus the shallower penetration',
       abs(ka.signed_gap(a, (1.5, -1.0, 4.0, 3.0)) + 0.5) < 1e-12)
    import numpy as np
    B = np.asarray([(5.0, 0.5, 6.0, 1.0), (5.0, 6.0, 7.0, 8.0), (1.5, -1.0, 4.0, 3.0)])
    ok('the vector form agrees', np.allclose(ka.signed_gaps(a, B), [3.0, 5.0, -0.5]))
    ok('the price: weight x shortfall^2, nothing past the gap',
       ka.shortfall_cost(3.0, 2.0, 10.0) == 10.0 and ka.shortfall_cost(3.0, 3.5, 10.0) == 0.0)
    ok('the price is capped DEEP_MM inside',
       ka.shortfall_cost(1.0, -50.0, 1.0) == (1.0 + ka.DEEP_MM) ** 2)


def _flipped(td):
    path = os.path.join(td, 'flip_esp_prog.kicad_pcb')
    if not os.path.exists(path):
        ps._face_board(BOARD, parse_kicad_pcb(BOARD), {r: 'test' for r in BACK}, td, 'flip',
                       BOARD)
    return path


def _model(td, rules=True):
    board = _flipped(td)
    pcb = parse_kicad_pcb(board)
    intent_path = os.path.join(td, 'intent.json')
    json.dump(fp.emit_intent(pcb, board), open(intent_path, 'w'))
    groups = os.path.join(td, 'groups.json')
    json.dump(GROUPS, open(groups, 'w'))
    rp = os.path.join(td, 'rules.json')
    json.dump(RULES, open(rp, 'w'))
    args = ps.build_parser().parse_args(
        [board, os.path.join(td, 'o.kicad_pcb'), '--groups', groups, '--intent', intent_path]
        + (['--keep-apart', rp, '--keep-apart-weight', '7'] if rules else []))
    intent = fp.load_intent(intent_path)
    blocks, _ = fp.resolve_blocks(intent, pcb, parse_sources('auto'))
    doc = json.load(open(intent_path))
    state, model, movable, _notes = ps.build(
        pcb, board, load_groups(groups), doc, args, intent, blocks, set())
    return state, model, sorted(movable)


def test_priced_locally_as_from_scratch(td):
    state, model, movable = _model(td)
    ok('the term is on', model.kap_on and len(model.kap.anchors) == 4,
       sorted(model.kap.anchors))
    ok('the back parts are on the back', all(state.parts[r].side == 'B' for r in BACK))
    rng = random.Random(11)
    x0, y0, x1, y1 = model.pcb.board_info.board_bounds
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    model.reprice_all()
    cur = model.total()['cost']
    worst, worst_pairs, priced, kept = 0.0, 0.0, 0, 0
    faces = set()
    for k in range(400):
        # one or two parts, crowded near the middle where the anchors are
        refs = rng.sample(movable, 2 if k % 5 == 0 else 1)
        if k % 3 == 0 and 'U2' in movable and 'U2' not in refs:
            refs = refs[:1] + ['U2']
        new = {r: (cx + rng.uniform(-9, 9), cy + rng.uniform(-6, 6),
                   rng.choice((0.0, 90.0, 180.0, 270.0))) for r in refs}
        refs = sorted(new)
        for r, (_x, _y, rot) in new.items():
            state.parts[r].ensure_rotation(rot)
        t = model.touched(refs)
        base = model.local(refs, t, reprice=False)
        snap = model.snapshot(t, refs)
        was = {r: (state.parts[r].x, state.parts[r].y, state.parts[r].rot) for r in refs}
        for r in refs:
            _place(state, r, *new[r])
        dc = model.local(refs, t) - base
        if rng.random() < 0.7:
            model.commit(refs, t[0])
            cur += dc
            kept += 1
            for (_i, a, b) in model._kpp:
                faces.add(state.parts[a].side == state.parts[b].side)
        else:
            for r in refs:
                _place(state, r, *was[r])
            model.restore(snap)
        if k % 25 == 24:
            committed = sum(model._kpp.values())
            scratch = model.keep_apart_cost()[0]
            worst_pairs = max(worst_pairs, abs(committed - scratch))
            tot = model.total()
            worst = max(worst, abs(tot['cost'] - cur))
            priced += tot['keep_apart'] > 0
            cur = tot['cost']
    ok('the moves broke rules', priced >= 5, priced)
    ok('pairs on one face and across the board were priced', faces == {True, False}, faces)
    ok('moves were kept', kept > 200, kept)
    ok('the committed pairs are the from-scratch term', worst_pairs < 1e-9, worst_pairs)
    ok('a move is priced locally as total() prices it', worst < 1e-6, worst)


def _run(td, tag, *extra):
    fixed = os.path.join(td, 'fixed.kicad_pcb')
    if not os.path.exists(fixed):
        subprocess.run([sys.executable, '-X', 'utf8', os.path.join(ROOT, 'py_placer', 'place_pose.py'),
                        BOARD, fixed, 'lock', 'USB1', 'CON1', 'CON2', 'Ref*', 'Ref*~2'],
                       capture_output=True, cwd=ROOT)
        subprocess.run([sys.executable, '-X', 'utf8', os.path.join(ROOT, 'py_tools', 'check_floorplan.py'),
                        fixed, '--emit-intent', os.path.join(td, 'intent.json')],
                       capture_output=True, cwd=ROOT)
        json.dump(GROUPS, open(os.path.join(td, 'groups.json'), 'w'))
        json.dump(RULES, open(os.path.join(td, 'rules.json'), 'w'))
    out, js = os.path.join(td, f'{tag}.kicad_pcb'), os.path.join(td, f'{tag}.json')
    r = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(ROOT, 'py_placer', 'place_simple.py'),
                        fixed, out, '--groups', os.path.join(td, 'groups.json'),
                        '--intent', os.path.join(td, 'intent.json'), '--seed', '1',
                        '--moves', '3000', '--json', js] + list(extra),
                       capture_output=True, text=True, cwd=ROOT)
    board = open(out, 'rb').read() if os.path.exists(out) else None
    rep = json.load(open(js)) if os.path.exists(js) else None
    return r, board, rep


def test_off_is_unchanged_on_is_reported(td):
    _r0, b0, j0 = _run(td, 'plain')
    _r1, b1, j1 = _run(td, 'weight', '--keep-apart-weight', '99')
    ok('both placed', b0 is not None and b1 is not None and j0 and j1)
    if not (b0 and b1 and j0 and j1):
        return
    ok('no file writes the same board', b0 == b1)
    ok('no file ends at the same cost', j0['final']['cost'] == j1['final']['cost'],
       (j0['final']['cost'], j1['final']['cost']))
    ok('no file reports no keep_apart', 'keep_apart' not in j1['final']
       and 'keep_apart_broken' not in j1['final'])
    rules = os.path.join(td, 'rules.json')
    r2, b2, j2 = _run(td, 'on', '--keep-apart', rules)
    ok('with rules placed', b2 is not None and j2 is not None, r2.stderr[-300:])
    if not j2:
        return
    ok('--json carries keep_apart and its broken pairs',
       'keep_apart' in j2['final'] and isinstance(j2['final'].get('keep_apart_broken'), list))
    summ = [ln for ln in r2.stdout.splitlines() if ln.startswith('JSON_SUMMARY: ')]
    ok('JSON_SUMMARY carries keep_apart',
       summ and 'keep_apart_broken' in json.loads(summ[-1][len('JSON_SUMMARY: '):])['final'])
    ok('the log line carries keep_apart', 'keep_apart ' in r2.stdout)
    ok('a note names the term', 'keep_apart priced: 2 rule(s), 4 anchor(s)' in r2.stdout)
    # the grader reads the written board as the anneal left it: at weight 0
    # the rules are measured and reported but not priced, so pairs stay broken
    r4, _b4, j4 = _run(td, 'zero', '--keep-apart', rules, '--keep-apart-weight', '0')
    ok('at weight 0 the pairs are reported, not priced',
       j4 is not None and j4['final']['keep_apart'] == 0.0
       and len(j4['final']['keep_apart_broken']) >= 2, (j4 or {}).get('final'))
    ok('... and the rules cost the placement something',
       len(j2['final']['keep_apart_broken']) < len(j4['final']['keep_apart_broken']),
       (j2['final']['keep_apart_broken'], j4['final']['keep_apart_broken']))
    out = os.path.join(td, 'grade.json')
    g = subprocess.run([sys.executable, '-X', 'utf8',
                        os.path.join(ROOT, 'py_tools', 'check_keep_apart.py'),
                        os.path.join(td, 'zero.kicad_pcb'), '--rules', rules,
                        '--groups', os.path.join(td, 'groups.json'),
                        '--intent', os.path.join(td, 'intent.json'), '--json', out],
                       capture_output=True, text=True, cwd=ROOT)
    rep = json.load(open(out)) if os.path.exists(out) else None
    ok('the grader ran', rep is not None and g.returncode in (0, 4), g.stderr[-300:])
    if rep is None:
        return
    mine = sorted((b[0], b[1], b[2]) for b in j4['final']['keep_apart_broken'])
    theirs = sorted((b[0], b[1], b[2]) for b in rep['broken'])
    ok('the grader breaks the pairs the anneal reports', mine == theirs, (mine, theirs))
    ok('... at the same distances', all(
        abs(x[3] - y[3]) < 2e-3 for x, y in zip(sorted(j4['final']['keep_apart_broken']),
                                              sorted(rep['broken']))))
    ok('the grader exits 4 on a break, else 0', g.returncode == (4 if theirs else 0))
    # a tight rule nobody can meet: the grader and the report both break it
    tight = json.loads(json.dumps(RULES))
    tight['rules'][1]['gap_mm'] = 30.0
    tp = os.path.join(td, 'tight.json')
    json.dump(tight, open(tp, 'w'))
    g2 = subprocess.run([sys.executable, '-X', 'utf8',
                         os.path.join(ROOT, 'py_tools', 'check_keep_apart.py'),
                         os.path.join(td, 'zero.kicad_pcb'), '--rules', tp,
                         '--groups', os.path.join(td, 'groups.json')],
                        capture_output=True, text=True, cwd=ROOT)
    ok('a rule no placement meets exits 4', g2.returncode == 4, g2.stdout[-300:])
    bad = os.path.join(td, 'bad.json')
    json.dump({'schema': 1, 'rules': [], 'x': 1}, open(bad, 'w'))
    r3 = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(ROOT, 'py_placer', 'place_simple.py'),
                         os.path.join(td, 'fixed.kicad_pcb'), os.path.join(td, 'bad.kicad_pcb'),
                         '--groups', os.path.join(td, 'groups.json'),
                         '--intent', os.path.join(td, 'intent.json'), '--keep-apart', bad],
                        capture_output=True, text=True, cwd=ROOT)
    ok('place_simple refuses a bad file at exit 2', r3.returncode == 2
       and 'unknown top-level key' in r3.stderr, r3.stderr[-300:])


def test_judge_ranks_broken_pairs_as_defects():
    def rep(cost, broken):
        return {'final': {'cost': cost, **({'keep_apart_broken': broken} if broken is not None
                                           else {})},
                'floorplan': {'errors': 0}, 'assembly': {'blocking': 0}}
    order = ps.judge_order({0: rep(10.0, [['r', 'A', 'B', 1.0]]), 1: rep(20.0, []),
                            2: rep(15.0, None)})
    ok('a seed with a broken pair ranks after the clean ones, cheaper or not',
       order == [2, 1, 0], order)


if __name__ == '__main__':
    test_validation()
    test_signed_gap()
    test_judge_ranks_broken_pairs_as_defects()
    with tempfile.TemporaryDirectory() as td:
        test_priced_locally_as_from_scratch(td)
    with tempfile.TemporaryDirectory() as td:
        test_off_is_unchanged_on_is_reported(td)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
