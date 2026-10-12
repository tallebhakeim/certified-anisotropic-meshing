"""Every number quoted in the JCE manuscript, computed with the exact parameters
of make_article_figures_en.py. Output: jce_numbers.txt (read by gen_latex_jce.py)."""
import json, numpy as np
import pointsource_certified as ps, structure_certified as sc
out = {}
# Fig 4: point source on film / substrate
h, s1, s2, e2, Rmax, a = 30e-9, 100.0, 1.0, 100e-6, 100e-6, 30e-9/4
P, zs, rs, v, src = ps.solve_axisym(h, e2, s1, s2, Rmax, a, nr=95, nz1=5, nz2=40)
surf = np.where(P[:, 1] < 1e-12)[0]; rr = P[surf, 0]; vv = v[surf]; o = np.argsort(rr); rr, vv = rr[o], vv[o]
out["ps_nodes"] = int(len(P)); out["ps_tris"] = int(2*(len(rs)-1)*(len(zs)-1))
out["ps_dev"] = {str(k): 100*(np.interp(k*h, rr, vv)/ps.analytic_Vsurf(k*h, h, s1, s2)-1) for k in [1, 2, 5, 10, 20, 30, 60]}
rc = np.logspace(0, 1.8, 40)*h; rc = rc[rc < 2.5e-6]
dev = np.array([100*(np.interp(x, rr, vv)/ps.analytic_Vsurf(x, h, s1, s2)-1) for x in rc])
out["ps_maxabsdev"] = float(np.abs(dev).max()); out["ps_rmax_over_h"] = float(rc[-1]/h)
dr = np.diff(rs); out["ps_aspect_max"] = float(dr[(rs[1:] < 3e-6)].max()/(h/5))
# Fig 6: stack, two cases
s_R1, s_R3 = 200.0, 21.6; d1, ei, d2, rcc, Rmx = 300e-9, 30e-9, 6e-6, 200e-9, 20e-6
for name, ssh in [("conductive", 1280.0), ("resistive", 2.16)]:
    P, zs, rs, v, src, b = sc.solve_stack([(d1, s_R1, 6), (ei, ssh, 4), (d2, s_R3, 40)], Rmx, rcc)
    axn = np.where(P[:, 0] < 1e-14)[0]; za = P[axn, 1]; va = v[axn]; o = np.argsort(za); za, va = za[o], va[o]
    Vt = np.interp(0, za, va); drop = np.interp(d1, za, va)-np.interp(d1+ei, za, va)
    out["stack_"+name] = dict(R=float(Vt), drop_pct=float(100*drop/Vt), nodes=int(len(P)))
# Fig 7 brackets: see jce_brackets.py (square contact on grid lines)
json.dump(out, open("jce_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
