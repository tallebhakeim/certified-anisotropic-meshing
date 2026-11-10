"""Two-sided bounds on a two-terminal resistance in 2D (per unit width).

Lower bound of R: P1 potential V_h (V = 1 on contact A, 0 on contact B), R_lo = 1 / int sigma |grad V_h|^2.
Upper bound of R: P1 stream function psi_h (psi = 0 and 1 on the two insulated boundary parts),
                  current q = (d psi/dz, -d psi/dx), R_up = int rho |grad psi_h|^2.
Both are plain P1 problems on the same triangles (structured anisotropic grid, cells split in two).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def graded(a, b, h_a, h_b, ratio=1.3):
    """Points from a to b; step h_a at a, h_b at b, growing geometrically by at most `ratio` in between."""
    L = b - a
    hmax = L
    # steps growing from each end until they meet
    left, right = [], []
    sl, sr, hl, hr = 0.0, 0.0, h_a, h_b
    while sl + sr + min(hl, hr) < L:
        if hl <= hr:
            left.append(hl); sl += hl; hl *= ratio
        else:
            right.append(hr); sr += hr; hr *= ratio
    steps = np.array(left + right[::-1])
    steps *= L / steps.sum()
    return a + np.concatenate([[0.0], np.cumsum(steps)])


def grid_mesh(xs, zs, keep):
    """Tensor grid, cell (i,j) kept if keep(xc, zc); each cell split in two triangles.
    Returns nodes (n,2), tris (m,3), centroid array."""
    nx, nz = len(xs), len(zs)
    idx = -np.ones((nx, nz), dtype=int)
    tris = []
    used = np.zeros((nx, nz), dtype=bool)
    cells = []
    for i in range(nx - 1):
        xc = 0.5 * (xs[i] + xs[i + 1])
        for j in range(nz - 1):
            zc = 0.5 * (zs[j] + zs[j + 1])
            if keep(xc, zc):
                cells.append((i, j))
                used[i:i + 2, j:j + 2] = True
    idx[used] = np.arange(used.sum())
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    nodes = np.column_stack([X[used], Z[used]])
    for (i, j) in cells:
        a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
        if (i + j) % 2 == 0:
            tris += [(a, b, c), (a, c, d)]
        else:
            tris += [(a, b, d), (b, c, d)]
    return nodes, np.array(tris)


def p1_energy(nodes, tris, coef, fixed):
    """Minimum of int coef |grad u|^2 over P1 functions with u = fixed[node] on the given nodes."""
    x, z = nodes[:, 0], nodes[:, 1]
    i, j, k = tris.T
    b = np.column_stack([z[j] - z[k], z[k] - z[i], z[i] - z[j]])
    c = np.column_stack([x[k] - x[j], x[i] - x[k], x[j] - x[i]])
    area2 = (x[j] - x[i]) * (z[k] - z[i]) - (x[k] - x[i]) * (z[j] - z[i])
    assert np.all(area2 > 0)
    Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) * (coef / (2.0 * area2))[:, None, None]
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    n = len(nodes)
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    u = np.zeros(n)
    fn = np.fromiter(fixed.keys(), dtype=int)
    u[fn] = np.fromiter(fixed.values(), dtype=float)
    free = np.setdiff1d(np.arange(n), fn)
    rhs = -(K[free][:, fn] @ u[fn])
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return float(np.dot(u, K.dot(u))), u


def enclosure(nodes, tris, sigma, nodes_A, nodes_B, nodes_N1, nodes_N2):
    """Return R_lo, R_up (per unit width) and the two discrete fields."""
    fixV = {int(n): 1.0 for n in nodes_A}
    fixV.update({int(n): 0.0 for n in nodes_B})
    G_up, V = p1_energy(nodes, tris, sigma, fixV)
    fixP = {int(n): 0.0 for n in nodes_N1}
    fixP.update({int(n): 1.0 for n in nodes_N2})
    R_up, psi = p1_energy(nodes, tris, 1.0 / sigma, fixP)
    return 1.0 / G_up, R_up, V, psi
