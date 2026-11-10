"""Hand example: step conductor (width 2 for 0<x<1, width 1 for 1<x<2, sigma = 1).
By hand: equipotential slices give R >= 1.5, straight current tubes give R <= 2.
The P1 potential / P1 stream function pair tightens the interval."""
import json
import numpy as np
from bounds2d import grid_mesh, enclosure, graded

out = []
for n in (4, 8, 16, 32, 64, 128):
    h = 1.0 / n
    xs = np.linspace(0, 2, 2 * n + 1)
    zs = np.linspace(0, 2, 2 * n + 1)
    keep = lambda xc, zc: (xc < 1) or (zc < 1)
    nodes, tris = grid_mesh(xs, zs, keep)
    x, z = nodes[:, 0], nodes[:, 1]
    e = 1e-12
    A = np.where(x < e)[0]                       # left end, V = 1
    B = np.where(x > 2 - e)[0]                   # right end, V = 0
    N1 = np.where(z < e)[0]                      # bottom side
    N2 = np.where(((z > 2 - e) & (x <= 1 + e)) | ((np.abs(x - 1) < e) & (z >= 1 - e)) | ((np.abs(z - 1) < e) & (x >= 1 - e)))[0]
    Rlo, Rup, _, _ = enclosure(nodes, tris, np.ones(len(tris)), A, B, N1, N2)
    out.append(dict(n=n, nodes=len(nodes), R_lo=Rlo, R_up=Rup, eta=(Rup - Rlo) / Rup))
    print(n, len(nodes), "%.5f %.5f  %.2f %%" % (Rlo, Rup, 100 * (Rup - Rlo) / Rup))
json.dump(out, open("jce_step.json", "w"), indent=1)
