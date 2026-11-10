"""Contact resistance of a planar contact on a thin semiconductor film (transfer-length structure), 2D.

Film: thickness t, sheet resistance Rsh. Contact: length L, specific contact resistivity rho_c, represented by an
interfacial layer of thickness delta and conductivity delta / rho_c under an equipotential metal.
Half structure: contact [0, L], half gap [L, L + g]; the mid-gap plane is an equipotential.
All resistances are per unit width (ohm m); printed in ohm um.
"""
import numpy as np
from bounds2d import grid_mesh, enclosure, graded


def tlm_1d(rho_c, Rsh, L):
    LT = np.sqrt(rho_c / Rsh)
    return rho_c / LT / np.tanh(L / LT)


def build(t, delta, L, g, level, family="graded"):
    """Structured grid. The smallest lateral step, at the contact edge, is t / (4 * 2**level).
    family "graded": the growth ratio of the steps decreases with the level, 1 + 0.4 / 2**level.
    family "edge":   the growth ratio stays at 1.4 (only the neighbourhood of the contact edge is refined).
    One cell across the interfacial layer in both families."""
    ratio = 1.0 + 0.4 / 2 ** level if family == "graded" else 1.4
    h0 = t / 4.0 / 2 ** level
    hfar = max(L, g) / 4.0 / 2 ** level
    xs = np.concatenate([graded(0.0, L, hfar, h0, ratio)[:-1], graded(L, L + g, h0, hfar, ratio)])
    nzf = 4 * 2 ** level
    zf = graded(0.0, t, 2.0 * t / nzf, 0.5 * t / nzf, ratio)
    zs = np.concatenate([zf, [t + delta]])
    keep = lambda xc, zc: (zc < t) or (xc < L)
    nodes, tris = grid_mesh(xs, zs, keep)
    return nodes, tris, xs, zs


def solve(t, Rsh, rho_c, delta, L, g, level, family="graded"):
    nodes, tris, xs, zs = build(t, delta, L, g, level, family)
    x, z = nodes[:, 0], nodes[:, 1]
    zc = z[tris].mean(axis=1)
    sigma = np.where(zc < t, 1.0 / (Rsh * t), delta / rho_c)
    e = 1e-9 * t
    top = t + delta
    A = np.where((z > top - e) & (x <= L + e))[0]
    B = np.where(x > L + g - e)[0]
    N1 = np.where((x < e) | (z < e))[0]
    N2 = np.where(((np.abs(x - L) < e) & (z >= t - e)) | ((np.abs(z - t) < e) & (x >= L - e)))[0]
    Rlo, Rup, V, psi = enclosure(nodes, tris, sigma, A, B, N1, N2)
    amax = np.max(np.diff(xs[xs <= L + e])) / delta   # widest interfacial-layer cell
    return dict(R_lo=Rlo, R_up=Rup, nodes=len(nodes), tris=len(tris), aspect=amax,
                mesh=(nodes, tris, V, psi, xs, zs))


if __name__ == "__main__":
    t, Rsh, delta, L, g = 50e-9, 100.0, 1e-9, 2e-6, 1e-6
    for rho_c_cm2 in (1e-6, 1e-7, 1e-8, 1e-9):
        rho_c = rho_c_cm2 * 1e-4
        print("rho_c = %.0e ohm cm2, LT = %.1f nm, TLM Rc = %.3f ohm um" % (rho_c_cm2, 1e9 * np.sqrt(rho_c / Rsh), 1e6 * tlm_1d(rho_c, Rsh, L)))
        for level in range(0, 6):
            r = solve(t, Rsh, rho_c, delta, L, g, level)
            lo, up = r["R_lo"] - Rsh * g, r["R_up"] - Rsh * g
            print("  level %d: %6d nodes, aspect %6.0f, Rc in [%.4f, %.4f] ohm um, width %.3f %%" % (level, r["nodes"], r["aspect"], 1e6 * lo, 1e6 * up, 100 * (up - lo) / up))
