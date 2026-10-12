"""Complementary (dual) capacitance lower bound via the RT0-P0 mixed FEM.

The primal nodal FEM (primal3d) gives an UPPER bound on the electrostatic
energy (Dirichlet principle).  This module gives the matching LOWER bound from
the complementary (Thomson) principle, discretized with the lowest-order
Raviart-Thomas flux element RT0 (one DoF per face) and a piecewise-constant
potential P0 (one DoF per tet) as Lagrange multiplier enforcing div D = 0.

Unlike the circumcentric "diagonal-Hodge" dual, this is a genuine conforming
H(div) discretization, so it gives a guaranteed lower bound on ANY tet mesh,
including at convex conductor boundaries.

RT0 basis (face opposite vertex i):  RT_i(x) = (x - p_i)/(3V), with unit flux
through face i, zero through the others, and div RT_i = 1/V.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _tet_faces_and_signs(mesh):
    """Per-tet global face indices (Nt,4) and orientation signs (Nt,4).

    Sign convention: the face DoF measures flux in the direction OUT of the
    tet listed first in mesh.face_tets; that tet gets +1, the other -1.
    """
    findex = {tuple(f): i for i, f in enumerate(mesh.faces.tolist())}
    gf = np.zeros((mesh.nt, 4), int)
    sgn = np.zeros((mesh.nt, 4))
    for t in range(mesh.nt):
        v = mesh.tets[t]
        for i_local in range(4):
            tri = tuple(sorted(int(v[j]) for j in range(4) if j != i_local))
            fi = findex[tri]
            gf[t, i_local] = fi
            sgn[t, i_local] = 1.0 if mesh.face_tets[fi][0][0] == t else -1.0
    return gf, sgn


def assemble_rt0(mesh, eps_tet):
    """Assemble RT0 mass matrix A (nf x nf) and div matrix B (nt x nf)."""
    eps_tet = np.broadcast_to(np.asarray(eps_tet, float), (mesh.nt,))
    gf, sgn = _tet_faces_and_signs(mesh)
    P = mesh.points[mesh.tets]                          # (Nt,4,3)
    V = mesh.volume
    c = P.mean(axis=1)                                  # centroid (Nt,3)
    # second moment term  s2 = (1/20) sum_k |p_k - c|^2
    s2 = (((P - c[:, None, :]) ** 2).sum(-1)).sum(-1) / 20.0   # (Nt,)
    # vertex opposite local face i is P[:, i]
    rows_a, cols_a, vals_a = [], [], []
    rows_b, cols_b, vals_b = [], [], []
    for t in range(mesh.nt):
        Vt, ept = V[t], eps_tet[t]
        ct = c[t]
        d = ct[None, :] - P[t]                          # (4,3)  c - p_i
        # A_ij = s_i s_j /(9 eps V) [ s2 + (c-p_i).(c-p_j) ]
        dot = d @ d.T                                   # (4,4)
        Ae = (s2[t] + dot) / (9.0 * ept * Vt)
        Ae = (sgn[t][:, None] * sgn[t][None, :]) * Ae
        for a in range(4):
            for b in range(4):
                rows_a.append(gf[t, a]); cols_a.append(gf[t, b]); vals_a.append(Ae[a, b])
            # div: B[t, face_a] = s_a * (flux of RT_a) = s_a  (∫ div RT_a = 1)
            rows_b.append(t); cols_b.append(gf[t, a]); vals_b.append(sgn[t, a])
    A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(mesh.nf, mesh.nf)).tocsr()
    B = sp.coo_matrix((vals_b, (rows_b, cols_b)), shape=(mesh.nt, mesh.nf)).tocsr()
    return A, B, gf, sgn


def capacitance_lower_3d(mesh, eps_tet, dirichlet, solver="direct"):
    """Complementary lower bound of the energy/capacitance (two-conductor).

    Parameters
    ----------
    dirichlet : dict {node: V} conductor potentials (used to set the natural
        BC on boundary faces whose 3 nodes share one conductor potential).

    Returns
    -------
    W_lower : complementary field energy (lower bound).
    C_lower : 2 * W_lower  (capacitance at unit voltage difference).
    D       : RT0 flux DoFs.

    The mesh is internally non-dimensionalised (scaled to unit extent) before the
    saddle-point solve — at e.g. micron coordinates the raw RT0 system is wildly
    ill-conditioned.  Capacitance is linear in length, so the result is scaled
    back by L afterwards.
    """
    from .mesh3d import TetMesh
    L = float(np.ptp(mesh.points, axis=0).max())       # characteristic length
    work = TetMesh(mesh.points / L, mesh.tets) if abs(L - 1.0) > 1e-9 else mesh
    A, B, gf, sgn = assemble_rt0(work, eps_tet)
    nf, nt = work.nf, work.nt

    # boundary faces: conductor (all nodes Dirichlet) -> natural BC g_f = -V_D ;
    # otherwise INSULATED (no-flux) -> essential BC, pin the flux DoF to zero.
    g = np.zeros(nf)
    pinned = np.zeros(nf, bool)
    for fi in range(nf):
        if not work.boundary_face[fi]:
            continue
        nodes = work.faces[fi]
        if all(int(n) in dirichlet for n in nodes):
            g[fi] = -np.mean([dirichlet[int(n)] for n in nodes])
        else:
            pinned[fi] = True                      # insulated wall: J.n = 0
    keep = ~pinned                                  # free flux DoFs
    A = A[keep][:, keep]; B = B[:, keep]; g = g[keep]; nf = int(keep.sum())

    if solver == "direct":
        # saddle-point system [[A, B^T],[B, 0]] [D; p] = [g; 0]
        K = sp.bmat([[A, B.T], [B, None]], format="csc")
        sol = spla.spsolve(K, np.concatenate([g, np.zeros(nt)]))
        D = sol[:nf]
    elif solver == "schur":
        # eliminate D = A^{-1}(g - B^T p); Schur S p = B A^{-1} g, S = B A^{-1} B^T
        # SPD => matrix-free conjugate gradient, A applied via one factorisation.
        lu = spla.splu(A.tocsc())
        Bc = B.tocsr(); BT = Bc.T.tocsr()
        S = spla.LinearOperator((nt, nt),
                                matvec=lambda p: Bc @ lu.solve(BT @ p))
        rhs = Bc @ lu.solve(g)
        p, info = spla.cg(S, rhs, rtol=1e-10, maxiter=2000)
        D = lu.solve(g - BT @ p)
    else:
        raise ValueError("solver must be 'direct' or 'schur'")
    W_lower = 0.5 * float(np.sum(D * (A @ D))) * L      # rescale: C ∝ length
    return W_lower, 2.0 * W_lower, D
