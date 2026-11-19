"""The group halo (`--group-halo-coef`): the front halo for a whole group.

Each group keeps, on each side of its envelope (the bounding box of its
members' courtyards, per face), a strip `coef x the nets leaving it by that
side` deep, and a part of no concern to the group inside the strip pays
`weight x shortfall^2` (`simple.Model._gh_*`).

Checked here, on esp_prog with three groups:

- a move is priced locally exactly as `total()` prices the board from
  scratch, over a few hundred random moves of single parts, pairs and whole
  groups (the anneal adds the local changes up, so a drift would steer it),
  and the committed (group, part) pairs add up to the from-scratch term;
- with the coefficient 0 (the default) the placer is unchanged: the same
  seed writes the same board and the same final cost as a run without the
  flags, and the breakdown carries no `group_halo`;
- with it on, the term is reported in the log's final line, JSON_SUMMARY and
  --json.
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
from placement.grouping import load_groups                    # noqa: E402
from placement.groups import parse_sources                    # noqa: E402
from placement.simple_anneal import _place                    # noqa: E402

BOARD = os.path.join(ROOT, 'kicad_files', 'esp_prog.kicad_pcb')
GROUPS = {'schema': 1, 'groups': [
    {'name': 'power', 'refs': ['U2', 'C1', 'C3']},
    {'name': 'core', 'refs': ['U1', 'Y1', 'C2', 'C4', 'R3', 'R4']},
    {'name': 'autoreset', 'refs': ['Q1', 'Q2', 'R1', 'R2']}]}
FAILS = []


def ok(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def _model(td, coef):
    pcb = parse_kicad_pcb(BOARD)
    intent_path = os.path.join(td, 'intent.json')
    json.dump(fp.emit_intent(pcb, BOARD), open(intent_path, 'w'))
    groups = os.path.join(td, 'groups.json')
    json.dump(GROUPS, open(groups, 'w'))
    args = ps.build_parser().parse_args(
        [BOARD, os.path.join(td, 'o.kicad_pcb'), '--groups', groups, '--intent', intent_path,
         '--group-halo-coef', str(coef)])
    intent = fp.load_intent(intent_path)
    blocks, _ = fp.resolve_blocks(intent, pcb, parse_sources('auto'))
    doc = json.load(open(intent_path))
    state, model, movable, _notes = ps.build(
        pcb, BOARD, load_groups(groups), doc, args, intent, blocks, set())
    return state, model, sorted(movable)


def test_priced_locally_as_from_scratch(td):
    state, model, movable = _model(td, 0.5)
    ok('the term is on', model.gh_on and len(model._ghg) == 3, len(model._ghg))
    rng = random.Random(7)
    x0, y0, x1, y1 = model.pcb.board_info.board_bounds
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    members = [[r for r in g['refs'] if r in movable] for g in GROUPS['groups']]
    model.reprice_all()
    cur = model.total()['cost']
    worst, worst_pairs, priced, kept = 0.0, 0.0, 0, 0
    for k in range(400):
        if k % 7 == 0:          # a whole group, shifted together
            refs = rng.choice(members)
            dx, dy = rng.uniform(-4, 4), rng.uniform(-4, 4)
            new = {r: (state.parts[r].x + dx, state.parts[r].y + dy, state.parts[r].rot)
                   for r in refs}
        else:                   # one or two parts, crowded near the middle
            refs = rng.sample(movable, 2 if k % 5 == 0 else 1)
            new = {r: (cx + rng.uniform(-8, 8), cy + rng.uniform(-6, 6),
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
        else:
            for r in refs:
                _place(state, r, *was[r])
            model.restore(snap)
        if k % 25 == 24:
            committed = sum(model._ghp.values())
            scratch = model.group_halo_cost()
            worst_pairs = max(worst_pairs, abs(committed - scratch))
            tot = model.total()
            worst = max(worst, abs(tot['cost'] - cur))
            priced += tot['group_halo'] > 0
            cur = tot['cost']
    ok('the moves came into the strips', priced >= 3, priced)
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
    _r1, b1, j1 = _run(td, 'zero', '--group-halo-coef', '0', '--group-halo-weight', '99')
    ok('both placed', b0 is not None and b1 is not None and j0 and j1)
    if not (b0 and b1 and j0 and j1):
        return
    ok('coef 0 writes the same board', b0 == b1)
    ok('coef 0 ends at the same cost', j0['final']['cost'] == j1['final']['cost'],
       (j0['final']['cost'], j1['final']['cost']))
    ok('coef 0 reports no group_halo', 'group_halo' not in j1['final'])
    r2, b2, j2 = _run(td, 'on', '--group-halo-coef', '0.5')
    ok('coef 0.5 placed', b2 is not None and j2 is not None, r2.stderr[-300:])
    if not j2:
        return
    ok('--json carries group_halo', 'group_halo' in j2['final'])
    summ = [ln for ln in r2.stdout.splitlines() if ln.startswith('JSON_SUMMARY: ')]
    ok('JSON_SUMMARY carries group_halo',
       summ and 'group_halo' in json.loads(summ[-1][len('JSON_SUMMARY: '):])['final'])
    ok('the log line carries group_halo', 'group_halo ' in r2.stdout)
    ok('a note names the term', 'group halo priced: 3 group(s)' in r2.stdout)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as td:
        test_priced_locally_as_from_scratch(td)
    with tempfile.TemporaryDirectory() as td:
        test_off_is_unchanged_on_is_reported(td)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
