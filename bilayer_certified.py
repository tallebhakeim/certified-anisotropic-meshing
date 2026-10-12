"""Certified conduction of a thin-film / substrate BILAYER on anisotropic cells.

Motivation (meeting 06/07/2026, Elomari Hall-effect study)
----------------------------------------------------------
Silvaco meshes the Hall test structure with near-equilateral triangles, so the
0.03 mm film on a 0.5 mm substrate over a 10 mm square forces the SAME tiny
in-plane step as the vertical one:  hundreds of thousands of nodes for one
layer, MILLIONS for two.  The thesis work-around exploits scale invariance
(results depend only on ratios D/S, e1/e2) to shrink everything to ~60 um, but
keeps the hard constraint "side > thickness" -> a truly thin film is still out
of reach.

Certified dual method (this file)
---------------------------------
The Hall measurement is, electrically, a steady conduction problem
    div(sigma grad V) = 0 ,  V pinned on the two current contacts.
The dual DGM brackets the two-terminal conductance G with a GUARANTEED pair:
  * primal P1 nodal FEM        -> upper bound on G  (Dirichlet principle)
  * RT0-P0 equilibrated mixed  -> lower bound on G  (Thomson principle)
so  R in [1/G_up, 1/G_lo]  with a certified width -- no reference solution.

Because the dual complementarity is robust to element stretching, we mesh the
film with a FLAT tensor grid: a few thin cells across the film, coarse cells
laterally.  Aspect ratios of 100:1-1000:1 cost a handful of cells where
Silvaco would need millions.  The certified width tells us when the coarse
anisotropic mesh is already good enough.

Benchmark:  a lateral bilayer resistor (full side-face contacts) has the exact
sheet conductance  G_exact = sigma1*e1 + sigma2*e2  (parallel sheets), which
must lie inside the certified bracket.
"""
from __future__ import annotations

import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgm.mesh3d import TetMesh
from dgm.primal3d import assemble_primal_3d
from dgm.mixed3d import capacitance_lower_3d


# ----------------------------------------------------------------------
# anisotropic tensor-grid -> tet mesh of a two-layer box
# ----------------------------------------------------------------------
# Kuhn/Freudenthal split of a unit cube into 6 tets (all share the main
# diagonal 0-7); reused for every hexahedron of the structured grid.
_KUHN = np.array([
    [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
    [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7],
])


def bilayer_mesh(W, e1, e2, nx, ny, nz1, nz2, Wy=None):
    """Two-layer box [0,Wx] x [0,Wy] x [0,e1+e2], film e1 on top of substrate e2.

    Wx = W; Wy defaults to W (square footprint).  Vertical grid is graded: nz2
    cells in the substrate (z in [0,e2]) and nz1 cells in the thin film
    (z in [e2, e2+e1]).  Lateral grid nx x ny is coarse, so the film cells are
    strongly stretched.

    Returns (mesh, layer_tet) with layer_tet=1 in the film, 0 in the substrate.
    """
    if Wy is None:
        Wy = W
    xs = np.linspace(0.0, W, nx + 1)
    ys = np.linspace(0.0, Wy, ny + 1)
    zs = np.concatenate([np.linspace(0.0, e2, nz2 + 1),
                         np.linspace(e2, e2 + e1, nz1 + 1)[1:]])
    nz = len(zs) - 1
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nX, nY, nZ = len(xs), len(ys), len(zs)

    def nid(i, j, k):
        return (i * nY + j) * nZ + k

    tets = []
    layer = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8 corners of the hex, local order matching _KUHN indexing
                c = [nid(i, j, k),         nid(i + 1, j, k),
                     nid(i, j + 1, k),     nid(i + 1, j + 1, k),
                     nid(i, j, k + 1),     nid(i + 1, j, k + 1),
                     nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                c = np.array(c)
                zc = 0.5 * (zs[k] + zs[k + 1])
                lay = 1 if zc > e2 else 0
                for t in _KUHN:
                    tets.append(c[t])
                    layer.append(lay)
    mesh = TetMesh(pts, np.array(tets))
    # TetMesh may drop degenerate tets; rebuild layer tag from tet centroids
    zc = mesh.points[mesh.tets][:, :, 2].mean(axis=1)
    layer_tet = (zc > e2).astype(int)
    return mesh, layer_tet


def _node_sets_xfaces(mesh, W, tol=1e-9):
    """Nodes on the x=0 and x=W faces (full side-face current contacts).

    Pure lateral flow -> the field is uniform in x and the dual pair is EXACT,
    so this is the exactness check: stretched cells add no error.
    """
    x = mesh.points[:, 0]
    a = np.where(x < tol)[0]
    b = np.where(x > W - tol)[0]
    return a, b


def _node_sets_toppads(mesh, W, e1, e2, frac=0.2, tol=1e-9):
    """Two current pads on the TOP surface z=e1+e2, x in [0,frac*W] and
    [(1-frac)*W, W], full y.  This is the realistic Hall/VdP contact layout:
    current is injected at the top film, spreads down into the substrate and
    back up -> a genuinely 3D field with vertical redistribution near the pads
    (the crowding that forces Silvaco to refine).  No closed form; the certified
    bracket is the only guaranteed error control.
    """
    z = mesh.points[:, 2]
    x = mesh.points[:, 0]
    top = z > (e1 + e2) - tol
    a = np.where(top & (x < frac * W + tol))[0]
    b = np.where(top & (x > (1.0 - frac) * W - tol))[0]
    return a, b


# ----------------------------------------------------------------------
# certified two-terminal conductance bracket
# ----------------------------------------------------------------------
def conductance_upper(mesh, sigma_tet, nodes_a, nodes_b):
    """Primal P1 upper bound on G for unit voltage between the two contacts."""
    K = assemble_primal_3d(mesh, sigma_tet)
    v = np.zeros(mesh.np)
    fixed = {int(n): 0.5 for n in nodes_a}
    fixed.update({int(n): -0.5 for n in nodes_b})
    fidx = np.array(sorted(fixed)); fval = np.array([fixed[i] for i in fidx])
    free = np.setdiff1d(np.arange(mesh.np), fidx)
    v[fidx] = fval
    rhs = -K[free][:, fidx] @ fval
    v[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    W_up = 0.5 * float(v @ (K @ v))
    return 2.0 * W_up, v                     # G_upper (unit voltage => G=2W)


def conductance_lower(mesh, sigma_tet, nodes_a, nodes_b):
    """RT0-P0 complementary lower bound on G."""
    dirichlet = {int(n): 0.5 for n in nodes_a}
    dirichlet.update({int(n): -0.5 for n in nodes_b})
    W_lo, _, _ = capacitance_lower_3d(mesh, sigma_tet, dirichlet)
    return 2.0 * W_lo                        # G_lower


def certified_bilayer(W, e1, e2, sigma1, sigma2, contacts="lateral",
                      nx=8, ny=8, nz1=2, nz2=4, verbose=False):
    """Certified two-terminal resistance of the bilayer.

    contacts="lateral" : full side-face contacts (exact benchmark).
    contacts="toppads"  : top-surface pads (realistic 3D field).
    """
    mesh, layer = bilayer_mesh(W, e1, e2, nx, ny, nz1, nz2)
    sigma_tet = np.where(layer == 1, sigma1, sigma2).astype(float)
    if contacts == "lateral":
        a, b = _node_sets_xfaces(mesh, W)
        g_exact = sigma1 * e1 + sigma2 * e2      # parallel sheet conductance
    else:
        a, b = _node_sets_toppads(mesh, W, e1, e2)
        g_exact = None                           # no closed form
    g_up, _ = conductance_upper(mesh, sigma_tet, a, b)
    g_lo = conductance_lower(mesh, sigma_tet, a, b)
    r_lo, r_hi = 1.0 / g_up, 1.0 / g_lo
    ref = r_hi                                   # scale for the width / test
    dz_film = e1 / nz1
    aspect = (W / nx) / dz_film
    if verbose:
        print(f"  ntet={mesh.nt:>6d}  aspect dx/dz_film={aspect:7.1f}")
        print(f"  G bracket [{g_lo:.6g}, {g_up:.6g}]"
              + (f"   G_exact={g_exact:.6g}" if g_exact else ""))
        print(f"  R bracket [{r_lo:.6g}, {r_hi:.6g}]"
              f"   certified width {max(0.0, (r_hi - r_lo) / ref) * 100:.3f}%")
        if g_exact is not None:
            r_exact = 1.0 / g_exact
            inside = r_lo <= r_exact * (1 + 1e-9) and r_exact <= r_hi * (1 + 1e-9)
            print(f"  R_exact={r_exact:.6g}   inside bracket: {inside}")
    return dict(ntet=mesh.nt, aspect=aspect, g=(g_lo, g_up), r=(r_lo, r_hi),
                g_exact=g_exact, width=(r_hi - r_lo) / ref)


if __name__ == "__main__":
    print("=" * 68)
    print("A. Non-dimensional benchmark: exact bilayer sheet conductance")
    print("=" * 68)
    # film 1/50 of substrate thickness, 1/500 of the lateral side
    W, e1, e2 = 1.0, 0.002, 0.10
    s1, s2 = 5.0, 1.0            # conductive film over resistive substrate
    print(f"W={W}  e1(film)={e1}  e2(sub)={e2}  sigma1={s1} sigma2={s2}")
    print(f"film aspect (W/e1) = {W / e1:.0f}\n")
    for (nx, nz1, nz2) in [(4, 1, 2), (8, 2, 4), (12, 2, 6), (16, 3, 8)]:
        print(f"grid nx=ny={nx}, nz_film={nz1}, nz_sub={nz2}:")
        certified_bilayer(W, e1, e2, s1, s2, contacts="lateral",
                          nx=nx, ny=nx, nz1=nz1, nz2=nz2, verbose=True)
        print()

    print("=" * 68)
    print("B. Realistic top pads: certified bracket tightens under refinement")
    print("   (genuine 3D field, vertical redistribution near the pads)")
    print("=" * 68)
    print(f"W={W}  e1={e1}  e2={e2}  film aspect W/e1={W/e1:.0f}\n")
    for (nx, nz1, nz2) in [(6, 1, 3), (10, 2, 4), (14, 2, 6), (18, 3, 8)]:
        print(f"grid nx=ny={nx}, nz_film={nz1}, nz_sub={nz2}:")
        certified_bilayer(W, e1, e2, s1, s2, contacts="toppads",
                          nx=nx, ny=nx, nz1=nz1, nz2=nz2, verbose=True)
        print()

    print("=" * 68)
    print("C. Thesis dimensions (the case Silvaco chokes on)")
    print("=" * 68)
    # 10 mm square, 0.5 mm substrate, 0.03 mm film  (metres)
    W, e1, e2 = 10e-3, 0.03e-3, 0.5e-3
    s1, s2 = 1e5, 1e3           # arbitrary two-conductivity contrast
    print(f"W={W*1e3:g}mm  e1={e1*1e6:g}um  e2={e2*1e3:g}mm  "
          f"film aspect W/e1={W/e1:.0f}\n")
    out = certified_bilayer(W, e1, e2, s1, s2, contacts="lateral",
                            nx=8, ny=8, nz1=2, nz2=4, verbose=True)
    # what an equilateral (Silvaco-style) mesh would cost at the film step
    dz = e1 / 2
    equilateral_nodes = (W / dz) ** 2 * ((e1 + e2) / dz)
    print(f"\n  cells used here: {out['ntet']}")
    print(f"  equilateral mesh at the film step dz={dz*1e6:g}um would need "
          f"~{equilateral_nodes:.2e} nodes")
    print(f"  reduction factor ~ {equilateral_nodes / out['ntet']:.1e}x")
