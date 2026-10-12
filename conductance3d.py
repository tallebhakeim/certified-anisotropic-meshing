"""Two-terminal certified conductance bounds on a tetrahedral mesh:
P1 primal upper bound and RT0-P0 complementary lower bound (from vdp_cross_3d.py)."""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgm.mesh3d import TetMesh


def conductance_upper_3d(mesh, sigma_tet, groups, a, b):
    """Primal P1 upper bound on G between contacts a,b; other contacts float."""
    from dgm.primal3d import assemble_primal_3d
    K = assemble_primal_3d(mesh, sigma_tet)
    npt = mesh.np
    master = np.arange(npt)
    for gi, g in enumerate(groups):
        if gi in (a, b) or len(g) == 0:
            continue
        master[g] = g[0]                               # tie floating contact
    uniq, inv = np.unique(master, return_inverse=True)
    P = sp.coo_matrix((np.ones(npt), (np.arange(npt), inv)),
                      shape=(npt, len(uniq))).tocsr()
    Kr = (P.T @ K @ P).tocsr()
    fixed = {}
    for n in groups[a]:
        fixed[inv[n]] = 0.5
    for n in groups[b]:
        fixed[inv[n]] = -0.5
    fidx = np.array(sorted(fixed)); fval = np.array([fixed[i] for i in fidx])
    free = np.setdiff1d(np.arange(Kr.shape[0]), fidx)
    v = np.zeros(Kr.shape[0]); v[fidx] = fval
    v[free] = spla.spsolve(Kr[free][:, free].tocsc(), -Kr[free][:, fidx] @ fval)
    W = 0.5 * float(v @ (Kr @ v))
    return 2.0 * W                                     # G_upper



def conductance_lower_3d(mesh, sigma_tet, groups, a, b):
    """Complementary (Thomson) RT0-P0 lower bound on G; other contacts float.

    Mirrors the 2D conductance_lower: conductor faces of a,b get the natural
    (Dirichlet) flux BC; every other boundary face is insulated (flux pinned to
    zero) EXCEPT the floating contacts, whose face fluxes are free but tied by a
    single zero-net-flux (zero net current) constraint each.  The mesh is scaled
    to unit extent first (G, like C, is linear in length) to tame conditioning.
    """
    from dgm.mixed3d import assemble_rt0
    L = float(np.ptp(mesh.points, axis=0).max())
    work = TetMesh(mesh.points / L, mesh.tets) if abs(L - 1.0) > 1e-9 else mesh
    A, B, gf, sgn = assemble_rt0(work, sigma_tet)
    nf, nt = work.nf, work.nt
    gset = [set(int(x) for x in g.tolist()) for g in groups]
    g_rhs = np.zeros(nf); pinned = np.zeros(nf, bool)
    float_faces = {gi: [] for gi in range(len(groups)) if gi not in (a, b)}
    for fi in range(nf):
        if not work.boundary_face[fi]:
            continue
        nodes = work.faces[fi]
        owner = None
        for gi, gs in enumerate(gset):
            if all(int(n) in gs for n in nodes):
                owner = gi
                break
        if owner == a:
            g_rhs[fi] = -0.5
        elif owner == b:
            g_rhs[fi] = +0.5
        elif owner is not None:
            float_faces[owner].append(fi)
        else:
            pinned[fi] = True
    keep = ~pinned
    idx = -np.ones(nf, int); idx[keep] = np.arange(int(keep.sum()))
    A2 = A[keep][:, keep]; B2 = B[:, keep]; g2 = g_rhs[keep]; nf2 = int(keep.sum())
    rows, cols, vals, nfl = [], [], [], 0
    for gi, fs in float_faces.items():
        if not fs:
            continue
        for fi in fs:
            rows.append(nfl); cols.append(idx[fi]); vals.append(1.0)
        nfl += 1
    blocks = [[A2, B2.T], [B2, None]]
    rhs = [g2, np.zeros(nt)]
    if nfl:
        C = sp.coo_matrix((vals, (rows, cols)), shape=(nfl, nf2)).tocsr()
        blocks = [[A2, B2.T, C.T], [B2, None, None], [C, None, None]]
        rhs.append(np.zeros(nfl))
    Ksys = sp.bmat(blocks, format="csc")
    sol = spla.spsolve(Ksys, np.concatenate(rhs))
    D = sol[:nf2]
    W = 0.5 * float(D @ (A2 @ D)) * L
    return 2.0 * W                                     # G_lower
