"""Numbers of the contact-resistance problem (Section 4): writes jce_contact.json.
Run: python3 jce_contact.py   (about 1 min)"""
import json
import numpy as np
from scipy.optimize import brentq
from tlm_contact import solve, tlm_1d

t, Rsh, delta, L, g = 50e-9, 100.0, 1e-9, 10e-6, 5e-6
UM = 1e6                                   # ohm m -> ohm um


def apparent_rho_c(Rc):
    """rho_c that the one-dimensional formula returns for a given contact resistance Rc (per unit width)."""
    return brentq(lambda r: tlm_1d(r, Rsh, L) - Rc, 1e-20, 1e-6, xtol=1e-30, rtol=1e-12)


out = dict(params=dict(t=t, Rsh=Rsh, delta=delta, L=L, g=g))

# 1. refinement at rho_c = 1e-8 ohm cm2, both mesh families
rho_c = 1e-8 * 1e-4
for fam in ("graded", "edge"):
    rows = []
    for level in range(0, 6):
        r = solve(t, Rsh, rho_c, delta, L, g, level, fam)
        lo, up = (r["R_lo"] - Rsh * g) * UM, (r["R_up"] - Rsh * g) * UM
        rows.append(dict(level=level, nodes=r["nodes"], tris=r["tris"], aspect=r["aspect"], Rc_lo=lo, Rc_up=up,
                         eta=100 * (up - lo) / up))
        print(fam, level, r["nodes"], "aspect %.0f" % r["aspect"], "[%.4f, %.4f] %.3f %%" % (lo, up, 100 * (up - lo) / up))
    for a, b in zip(rows, rows[1:]):
        assert b["Rc_lo"] >= a["Rc_lo"] - 1e-9 and b["Rc_up"] <= a["Rc_up"] + 1e-9, "not nested"
    out[fam] = rows

# 2. sweep of the specific contact resistivity, level 4 of the graded family
sweep = []
for rc_cm2 in (1e-6, 1e-7, 1e-8, 1e-9):
    rho_c = rc_cm2 * 1e-4
    r = solve(t, Rsh, rho_c, delta, L, g, 4, "graded")
    lo, up = r["R_lo"] - Rsh * g, r["R_up"] - Rsh * g
    one_d = tlm_1d(rho_c, Rsh, L)
    sweep.append(dict(rho_c_cm2=rc_cm2, LT_nm=1e9 * np.sqrt(rho_c / Rsh), t_over_LT=t / np.sqrt(rho_c / Rsh),
                      Rc_1d=one_d * UM, Rc_lo=lo * UM, Rc_up=up * UM, eta=100 * (up - lo) / up,
                      dev_lo=100 * (lo / one_d - 1), dev_up=100 * (up / one_d - 1),
                      app_lo=apparent_rho_c(lo) / rho_c, app_up=apparent_rho_c(up) / rho_c, nodes=r["nodes"]))
    print(sweep[-1])
out["sweep"] = sweep

# 2b. finer sweep for the figure (level 3)
curve = []
for rc_cm2 in np.logspace(-6, -9, 13):
    rho_c = rc_cm2 * 1e-4
    r = solve(t, Rsh, rho_c, delta, L, g, 3, "graded")
    lo, up = r["R_lo"] - Rsh * g, r["R_up"] - Rsh * g
    one_d = tlm_1d(rho_c, Rsh, L)
    curve.append(dict(rho_c_cm2=rc_cm2, t_over_LT=t / np.sqrt(rho_c / Rsh), dev_lo=100 * (lo / one_d - 1),
                      dev_up=100 * (up / one_d - 1), app_lo=apparent_rho_c(lo) / rho_c, app_up=apparent_rho_c(up) / rho_c))
out["curve"] = curve

# 3. thickness of the interfacial layer at fixed rho_c (1e-8 ohm cm2)
dl = []
for d in (0.5e-9, 1e-9, 2e-9):
    r = solve(t, Rsh, 1e-12, d, L, g, 4, "graded")
    dl.append(dict(delta_nm=d * 1e9, Rc_lo=(r["R_lo"] - Rsh * g) * UM, Rc_up=(r["R_up"] - Rsh * g) * UM))
    print(dl[-1])
out["delta"] = dl

# 4. node count of a mesh of near-square cells that puts one cell across the interfacial layer
out["iso_nodes_2d"] = (L + g) / delta * (t + delta) / delta
out["iso_nodes_3d_W10um"] = out["iso_nodes_2d"] * 10e-6 / delta
json.dump(out, open("jce_contact.json", "w"), indent=1)
