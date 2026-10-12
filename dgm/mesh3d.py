"""3D primal/dual mesh container for the Discrete Geometric Method.

Mirrors mesh2d.py one dimension up:
  - primal  : tetrahedra (Delaunay)
  - dual    : circumcentric (Voronoi), nodes at tet circumcenters
  - the dual edge crossing a primal triangular FACE connects the circumcenters
    of the two tets sharing that face, perpendicular to it.

Geometry conventions:
  - A_face : area of a primal triangular face.
  - l_D    : SIGNED distance from a tet circumcenter to one of its faces,
             positive when the circumcenter lies on the same side as the
             opposite vertex (interior side).  Kept signed, exactly as in 2D.
"""
from __future__ import annotations

import numpy as np


def _circumcenters(P):
    """Circumcenters of tets. P: (Nt,4,3). Returns (Nt,3)."""
    p0 = P[:, 0]
    M = P[:, 1:] - p0[:, None, :]                      # (Nt,3,3) rows = pi-p0
    b = 0.5 * (np.einsum("tij,tij->ti", P[:, 1:], P[:, 1:])
               - np.einsum("ti,ti->t", p0, p0)[:, None])
    c_rel = np.linalg.solve(M, b[..., None])[..., 0]   # (Nt,3)
    return p0 + c_rel


class TetMesh:
    """Primal tetrahedral mesh with derived circumcentric dual geometry."""

    def __init__(self, points, tets, vol_tol=1e-10):
        self.points = np.asarray(points, float)
        tets = np.asarray(tets, int)
        # drop (near-)degenerate tets: a regular Delaunay can emit flat slivers
        # whose circumcenter is undefined.  They carry no volume/energy anyway.
        P = self.points[tets]
        vol = np.abs(np.linalg.det(P[:, 1:] - P[:, 0:1])) / 6.0
        good = vol > vol_tol * vol.max()
        self.n_dropped = int((~good).sum())
        self.tets = tets[good]
        self.np = len(self.points)
        self.nt = len(self.tets)
        self._build_geometry()
        self._build_faces()

    def _build_geometry(self):
        P = self.points[self.tets]                     # (Nt,4,3)
        M = P[:, 1:] - P[:, 0:1]
        self.volume = np.abs(np.linalg.det(M)) / 6.0
        self.circumcenter = _circumcenters(P)

    def _local_face_geometry(self, t, i_local):
        """(A_face, signed l_D) for the face of tet t opposite local vertex i_local."""
        v = self.tets[t]
        opp = v[i_local]
        face = [v[j] for j in range(4) if j != i_local]
        q0, q1, q2 = (self.points[face[0]], self.points[face[1]],
                      self.points[face[2]])
        n = np.cross(q1 - q0, q2 - q0)
        area = 0.5 * np.linalg.norm(n)
        nhat = n / np.linalg.norm(n)
        if np.dot(self.points[opp] - q0, nhat) < 0:
            nhat = -nhat                               # orient toward opp vertex
        ld = np.dot(self.circumcenter[t] - q0, nhat)   # signed distance
        return area, ld

    def _build_faces(self):
        fdict = {}
        face_tets = []
        for t, v in enumerate(self.tets):
            for i_local in range(4):
                tri = tuple(sorted(int(v[j]) for j in range(4) if j != i_local))
                if tri not in fdict:
                    fdict[tri] = len(face_tets)
                    face_tets.append([])
                face_tets[fdict[tri]].append((t, i_local))
        self.faces = np.array(list(fdict.keys()), dtype=int)
        self.nf = len(self.faces)
        self.face_tets = face_tets
        self.boundary_face = np.array([len(ft) == 1 for ft in face_tets])

    def boundary_nodes(self):
        nodes = set()
        for f, is_b in zip(self.faces, self.boundary_face):
            if is_b:
                nodes.update(f.tolist())
        return np.array(sorted(nodes), dtype=int)


def voxel_to_tetmesh(sigma, h=1.0, mask=None):
    """Conforming tet mesh of a voxel block (6 tets per voxel, Kuhn split).

    sigma : (nz, ny, nx) per-voxel coefficient. Returns (mesh, coeff_tet) where
    coeff_tet[t] is the voxel value of tet t — enabling the validated primal +
    RT0 complementary bracket directly on segmented/voxel geometry.

    mask : optional (nz, ny, nx) bool — if given, ONLY the cells where mask is True
    are meshed (e.g. body voxels, excluding air), so the boundary surface is the
    actual anatomy rather than the bounding box.  Nodes are compacted to those used.
    """
    sigma = np.asarray(sigma, float)
    nz, ny, nx = sigma.shape
    gx, gy, gz = nx + 1, ny + 1, nz + 1
    Z, Y, X = np.mgrid[0:gz, 0:gy, 0:gx].astype(float)
    points = np.c_[X.ravel() * h, Y.ravel() * h, Z.ravel() * h]
    vid = np.arange(gx * gy * gz).reshape(gz, gy, gx)
    # 8 corner offsets (z,y,x), ordered 0..7 as binary (x fastest)
    off = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
           (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
            (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    zz, yy, xx = zz.ravel(), yy.ravel(), xx.ravel()
    sig_flat = sigma.ravel()
    if mask is not None:
        keep = np.asarray(mask, bool).ravel()
        zz, yy, xx, sig_flat = zz[keep], yy[keep], xx[keep], sig_flat[keep]
    corners = [vid[zz + dz, yy + dy, xx + dx] for (dz, dy, dx) in off]  # 8 x Ncell
    tets = []
    for (a, b, c, d) in kuhn:
        tets.append(np.c_[corners[a], corners[b], corners[c], corners[d]])
    tets = np.concatenate(tets, axis=0)
    coeff = np.tile(sig_flat, len(kuhn))
    if mask is not None:                                   # compact to used nodes
        used = np.unique(tets)
        remap = np.full(len(points), -1, int); remap[used] = np.arange(len(used))
        points = points[used]; tets = remap[tets]
    mesh = TetMesh(points, tets)
    # TetMesh may drop degenerate tets; coeff aligned because none are degenerate here
    return mesh, coeff


def _fibonacci_sphere(n, radius):
    """n roughly-uniform points on a sphere of given radius (Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return radius * np.c_[x, y, z]


def multilayer_sphere_mesh(radii, n_in=150, per_layer=4, seed=1):
    """Concentric multilayer spherical phantom (e.g. brain / skull / skin).

    Parameters
    ----------
    radii : list [r0, r1, ..., rN]  — r0 = inner electrode, rN = outer electrode,
        the intermediate values are the layer interfaces.
    n_in : surface points on the innermost sphere (density grows like r).
    per_layer : radial shells per layer (interfaces are always meshed).

    Returns
    -------
    mesh : TetMesh
    inner_nodes, outer_nodes : boundary node indices on r0 and rN.
    layer_of_tet : (Nt,) int — layer index (0..N-1) of each tet (by centroid r).
    """
    rng = np.random.default_rng(seed)
    r0, rN = radii[0], radii[-1]
    shell_r = []
    for i in range(len(radii) - 1):
        a, b = radii[i], radii[i + 1]
        for k in range(per_layer):
            shell_r.append(a + (b - a) * k / per_layer)
    shell_r.append(rN)
    shell_r = sorted(set(shell_r))
    pts = []
    for j, r in enumerate(shell_r):
        m = int(round(n_in * (r / r0)))
        sp = _fibonacci_sphere(m, 1.0)
        if 0 < j < len(shell_r) - 1:
            r = r * (1 + 0.03 * rng.standard_normal(m))[:, None]
        pts.append(sp * r)
    P = np.vstack(pts)
    from scipy.spatial import Delaunay
    tri = Delaunay(P)
    cen = P[tri.simplices].mean(axis=1)
    rc = np.linalg.norm(cen, axis=1)
    keep = (rc > r0) & (rc < rN)
    tets = tri.simplices[keep]
    used = np.unique(tets); remap = -np.ones(len(P), int); remap[used] = np.arange(len(used))
    P = P[used]; tets = remap[tets]
    mesh = TetMesh(P, tets)
    rr = np.linalg.norm(P, axis=1)
    inner = np.where(np.abs(rr - r0) < 1e-4 * rN)[0]
    outer = np.where(np.abs(rr - rN) < 1e-4 * rN)[0]
    cen = mesh.points[mesh.tets].mean(axis=1); rc = np.linalg.norm(cen, axis=1)
    layer = np.zeros(mesh.nt, int)
    for i in range(len(radii) - 1):
        layer[(rc >= radii[i]) & (rc < radii[i + 1])] = i
    return mesh, inner, outer, layer


def sphere_shell_mesh(r_in, r_out, n_in=160, n_out=320, n_shells=8, seed=0):
    """Delaunay tet mesh of the shell r_in < r < r_out.

    Returns
    -------
    mesh : TetMesh
    inner_nodes, outer_nodes : boundary node index arrays on each sphere.
    """
    rng = np.random.default_rng(seed)
    # logarithmic radial grading (dense near r_in where a 1/r^2 field is strong);
    # points per shell scale with r^2 so the surface density stays uniform.
    pts = []
    sr_in, sr_out = np.sqrt(r_in), np.sqrt(r_out)
    for k in range(n_shells + 1):
        # sqrt-spaced radii: makes radial spacing ~ tangential spacing (which
        # scales as sqrt(r) for m proportional to r), giving near-isotropic
        # tets and well-centred circumcenters (l_D > 0) up to the boundary.
        r = (sr_in + (sr_out - sr_in) * k / n_shells) ** 2
        m = int(round(n_in * (r / r_in)))
        sp = _fibonacci_sphere(m, 1.0)
        if 0 < k < n_shells:
            r = r * (1 + 0.04 * rng.standard_normal(m))[:, None]
        pts.append(sp * r)
    points = np.vstack(pts)

    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    cent = points[tri.simplices].mean(axis=1)
    rc = np.linalg.norm(cent, axis=1)
    keep = (rc > r_in) & (rc < r_out)
    tets = tri.simplices[keep]

    used = np.unique(tets)
    remap = -np.ones(len(points), int); remap[used] = np.arange(len(used))
    points = points[used]; tets = remap[tets]

    mesh = TetMesh(points, tets)
    rr = np.linalg.norm(points, axis=1)
    inner = np.where(np.abs(rr - r_in) < 1e-6 * r_out)[0]
    outer = np.where(np.abs(rr - r_out) < 1e-6 * r_out)[0]
    return mesh, inner, outer
