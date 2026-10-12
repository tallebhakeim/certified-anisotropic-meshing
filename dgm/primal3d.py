"""Primal 3D DGM = linear-tetrahedron nodal FEM (upper energy bound).

As in 2D, the primal DGM coincides with the Galerkin nodal FEM. For a linear
tet the shape-function gradients are constant, so
    K_e[i,j] = eps * V * (grad lambda_i . grad lambda_j).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


def element_gradients(mesh):
    """Per-tet shape-function gradients (Nt,4,3) and volumes (Nt,).

    For a linear tet the gradients of the barycentric coordinates are constant.
    Shared by the stiffness assembly and the Joule-power computation.
    """
    P = mesh.points[mesh.tets]                         # (Nt,4,3)
    J = (P[:, 1:] - P[:, 0:1]).transpose(0, 2, 1)      # (Nt,3,3) cols = pi-p0
    V = np.abs(np.linalg.det(J)) / 6.0
    Jinv = np.linalg.inv(J)
    grads = np.zeros((mesh.nt, 4, 3))
    grads[:, 1:, :] = Jinv                              # rows of J^{-1}
    grads[:, 0, :] = -Jinv.sum(axis=1)
    return grads, V


def assemble_primal_3d(mesh, coeff_tet):
    """Assemble the 3D nodal stiffness for div(coeff grad u) (Np x Np), sparse.

    coeff_tet : (Nt,) coefficient per tetrahedron — permittivity eps for the
        electrostatic/capacitance problem, electrical conductivity sigma for the
        current-conduction problem, or thermal conductivity k for heat.
    """
    coeff_tet = np.broadcast_to(np.asarray(coeff_tet), (mesh.nt,))   # keep dtype (complex OK)
    grads, V = element_gradients(mesh)
    eps_tet = coeff_tet
    # K_e[t,i,j] = coeff V <grad_i, grad_j>
    Ke = (eps_tet * V)[:, None, None] * np.einsum("tik,tjk->tij", grads, grads)

    rows = np.repeat(mesh.tets, 4, axis=1).reshape(mesh.nt, 4, 4)
    cols = np.tile(mesh.tets, (1, 4)).reshape(mesh.nt, 4, 4)
    K = sp.coo_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(mesh.np, mesh.np)).tocsr()
    return K
