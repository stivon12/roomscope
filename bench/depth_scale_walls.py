"""Held-out depth scale for the MuSHRoom iPhone: opposite-wall distances in our uncorrected vr_room runs vs the
Faro laser. Laser wall = mode of room-facing laser points within 0.3 m of our edge, 0.3-2.0 m above the floor.
vr_room is not one of the head-to-head rooms, so the value is fitted without them.

    python bench/depth_scale_walls.py [RUN_DIR]   # RUN_DIR/<capture>/*/result.json from runs with --no-depth-correction
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, 'bench')
from head_to_head import registered, crop, DATA
from mushroom_eval import load_faro
S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/bench/depth_scale_walls"); ratios = []
for cap in ['vr_room_long', 'vr_room_short']:
    d = next((S / cap).glob('*/result.json')).parent
    res = json.load(open(d / 'result.json')); cl = np.load(d / 'cloud.npz')
    P, N = cl['P'].astype(float), cl['N'].astype(float)
    pc, fit = registered(load_faro(DATA / 'vr_room/gt_pd.ply'), P, N); pc = crop(pc, res, P)
    L, LN = np.asarray(pc.points), np.asarray(pc.normals)
    fz = np.percentile(L[LN[:, 2] > 0.95, 2], 10)
    for rm in res['rooms']:
        poly = np.array(rm['polygon']); n = len(poly); edges = []
        for k in range(n):
            p, q = poly[k], poly[(k + 1) % n]; t = (q - p) / np.linalg.norm(q - p)
            axis = 0 if abs(t[0]) < abs(t[1]) else 1; inward = np.array([-t[1], t[0], 0.])
            lo, hi = sorted((p[1 - axis], q[1 - axis]))
            if hi - lo < 1.0: continue
            m = (LN @ inward > 0.95) & (np.abs(L[:, axis] - p[axis]) < 0.3) & (L[:, 1 - axis] > lo + .1) \
                & (L[:, 1 - axis] < hi - .1) & (L[:, 2] > fz + .3) & (L[:, 2] < fz + 2.0)
            if m.sum() < 500: continue
            h, e = np.histogram(L[m, axis], np.arange(p[axis] - .3, p[axis] + .31, .01))
            ref = e[h.argmax()] + .005; ref = L[m, axis][np.abs(L[m, axis] - ref) < .015].mean()
            edges.append((axis, int(np.sign(inward[axis])), p[axis], ref, lo, hi))
        for a, b in [(a, b) for i, a in enumerate(edges) for b in edges[i + 1:]]:
            if a[0] == b[0] and a[1] != b[1] and min(a[5], b[5]) - max(a[4], b[4]) > 0.5:
                ours, las = abs(a[2] - b[2]), abs(a[3] - b[3])
                ratios.append(ours / las); print(f'{cap}: width ours {ours:.3f} laser {las:.3f} ratio {ours/las:.4f}  (icp {fit:.2f})')
r = np.array(ratios)
print(f'n={len(r)} median ratio {np.median(r):.4f} mean {r.mean():.4f} sd {r.std(ddof=1) if len(r)>1 else 0:.4f}')
