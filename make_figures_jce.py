"""Figures of the J. Comput. Electron. version, at final print size
(84 mm single column, 174 mm double column), sans-serif, no title inside the
image (captions live in the manuscript). Vector PDF + EPS in figures_jce/.
Also writes jce_aspect.json (aspect-ratio sweep of the parallel-sheet benchmark)."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/Users/talleb/Desktop/travaux aboubaker/hall_certified_poc")
import bilayer_certified as bc
import pointsource_certified as ps, structure_certified as sc, certified_bracket as cb

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42})
MM = 1/25.4; COL1, COL2 = 84*MM, 174*MM
N = json.load(open("jce_numbers.json"))

def save(fig, name):
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "eps"): fig.savefig(f"figures_jce/{name}.{ext}")
    plt.close(fig)

# Fig 1: thin layer in a large body
fig, ax = plt.subplots(figsize=(COL2, 1.55))
ax.add_patch(Rectangle((0, 0), 10, 1.6, fc="#d8dee6", ec="k", lw=0.6))
ax.add_patch(Rectangle((0, 1.6), 10, 0.06, fc="#2f8f7f", ec="none"))
ax.annotate("", xy=(10.3, 1.57), xytext=(10.3, 1.69), arrowprops=dict(arrowstyle="<->", color="#c0601b", lw=0.7))
ax.text(10.5, 1.63, r"$e_1 \approx$ 10–100 nm", color="#c0601b", va="center")
ax.annotate("", xy=(0, -0.3), xytext=(10, -0.3), arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(5, -0.62, r"$W \approx$ 0.1–10 mm,   aspect ratio $A = W/e_1 \approx 10^3$–$10^5$", ha="center")
ax.text(5, 0.8, "body / substrate", ha="center", color="#444")
ax.text(5, 1.95, "thin layer (drawn to scale)", ha="center", color="#2f8f7f")
ax.set_xlim(-0.3, 14.3); ax.set_ylim(-0.9, 2.3); ax.axis("off")
save(fig, "Fig1")

# Fig 2: isotropic vs anisotropic discretization
fig, axs = plt.subplots(1, 2, figsize=(COL2, 1.75))
for ax, mode in zip(axs, ["iso", "aniso"]):
    ax.add_patch(Rectangle((0, 0), 10, 3, fc="#eef1f5", ec="k", lw=0.6))
    ax.add_patch(Rectangle((0, 2.6), 10, 0.4, fc="#d7f0ea", ec="none"))
    if mode == "iso":
        for x in np.arange(0, 10.01, 0.4): ax.plot([x, x], [0, 3], color="#c33", lw=0.25)
        for y in np.arange(0, 3.01, 0.4): ax.plot([0, 10], [y, y], color="#c33", lw=0.25)
        ax.text(5, -0.45, r"(a) isotropic: in-plane step $\approx e_1$, $N \propto A^2$", ha="center")
    else:
        for x in np.arange(0, 10.01, 1.25): ax.plot([x, x], [0, 3], color="#2a7", lw=0.45)
        for y in [0, 0.7, 1.4, 2.1, 2.6, 2.73, 2.87, 3.0]: ax.plot([0, 10], [y, y], color="#2a7", lw=0.45)
        ax.text(5, -0.45, r"(b) anisotropic: flat layer cells, $N$ independent of $A$", ha="center")
    ax.text(5, 2.75, "layer", ha="center", color="#2f8f7f", fontsize=7)
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.8, 3.2); ax.axis("off")
save(fig, "Fig2")

# Fig 3: aspect-ratio sweep, parallel-sheet benchmark (exact G = s1 e1 + s2 e2)
W, e1, e2, s1, s2 = 1.0, 0.002, 0.10, 5.0, 1.0
sweep = []
for nx, nz1, nz2 in [(50, 1, 2), (25, 1, 2), (10, 1, 2), (5, 1, 4), (4, 2, 4), (8, 2, 4), (4, 4, 6), (2, 4, 6)]:
    o = bc.certified_bilayer(W, e1, e2, s1, s2, contacts="lateral", nx=nx, ny=nx, nz1=nz1, nz2=nz2)
    rex = 1/o["g_exact"]; rlo, rhi = o["r"]
    sweep.append(dict(nx=nx, nz1=nz1, nz2=nz2, ntet=int(o["ntet"]), aspect=float(o["aspect"]),
                      gap=float(abs(rhi-rlo)/rhi), inside=bool(rlo <= rex*(1+1e-12) and rex <= rhi*(1+1e-12)),
                      relerr_up=float(abs(rlo-rex)/rex)))
sweep.sort(key=lambda d: d["aspect"]); json.dump(sweep, open("jce_aspect.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(COL1, 2.3))
asp = [d["aspect"] for d in sweep]
ax.loglog(asp, [max(d["gap"], 1e-16) for d in sweep], "o-", color="#2a7", ms=3.5, label=r"certified width $\eta$")
ax.axhline(2.2e-16, color="k", ls=":", lw=0.7); ax.text(asp[0], 4e-16, "double-precision epsilon", fontsize=6.5)
ax.set_xlabel(r"layer cell aspect ratio $\Delta x / \Delta z$"); ax.set_ylabel(r"certified width $\eta$")
ax.set_ylim(1e-17, 1e-2); ax.grid(alpha=0.3, which="major"); ax.legend(loc="upper right")
save(fig, "Fig3")

# Fig 4: point source on film / substrate
h, s1, s2, e2, Rmax, a = 30e-9, 100.0, 1.0, 100e-6, 100e-6, 30e-9/4
P, zs, rs, v, src = ps.solve_axisym(h, e2, s1, s2, Rmax, a, nr=95, nz1=5, nz2=40)
surf = np.where(P[:, 1] < 1e-12)[0]; rr = P[surf, 0]; vv = v[surf]; o = np.argsort(rr); rr, vv = rr[o], vv[o]
mask = (rr > 0) & (rr < 3e-6)
fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.4))
ra = np.logspace(np.log10(h), np.log10(3e-6), 200)
ax[0].loglog(ra/h, [ps.analytic_Vsurf(x, h, s1, s2) for x in ra], "k-", lw=1.2, label="closed form, Eq. (10)")
ax[0].loglog(rr[mask]/h, vv[mask], "o", ms=2.5, color="tab:blue", mfc="none", mew=0.6, label="P1 solution, anisotropic cells")
ax[0].set_xlabel(r"$r / e_1$"); ax[0].set_ylabel(r"surface potential $V(r)$ (V, $I$ = 1 A)")
ax[0].legend(); ax[0].grid(alpha=0.3, which="both"); ax[0].text(0.03, 0.05, "(a)", transform=ax[0].transAxes)
rc = np.logspace(0, 1.8, 40)*h; rc = rc[rc < 2.5e-6]
off = ps.analytic_Vsurf(Rmax, h, s1, s2)   # closed form at the grounded outer boundary
rel = [100*(np.interp(x, rr, vv)/ps.analytic_Vsurf(x, h, s1, s2)-1) for x in rc]
relc = [100*(np.interp(x, rr, vv)/(ps.analytic_Vsurf(x, h, s1, s2)-off)-1) for x in rc]
ax[1].semilogx(rc/h, rel, "o-", color="tab:green", ms=2.5, label=r"vs $V(r)$")
ax[1].semilogx(rc/h, relc, "s-", color="tab:gray", ms=2.2, label=r"vs $V(r) - V(R_{max})$")
ax[1].legend(loc="lower left", bbox_to_anchor=(0.10, 0.02))
ax[1].axhline(0, color="k", ls="--", lw=0.7); ax[1].set_ylim(-4, 1)
ax[1].set_xlabel(r"$r / e_1$"); ax[1].set_ylabel("deviation from closed form (%)")
ax[1].grid(alpha=0.3, which="both"); ax[1].text(0.03, 0.05, "(b)", transform=ax[1].transAxes)
save(fig, "Fig4")

# Fig 5: equilibrium band diagram (solver of band_diagram.py)
import band_diagram as bd
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
for ax, (Nsh, lab) in zip(axs, [(1e17, r"(a) sheet $N_D = 10^{17}$ cm$^{-3}$"), (1e14, r"(b) sheet $N_D = 10^{14}$ cm$^{-3}$")]):
    zs, phi, (zb, zc) = bd.solve_band(1e16, Nsh, 1e15)
    Ec = bd.EC0-phi; Ev = Ec-bd.EG; z = (zs-zb)*1e9
    ax.plot(z, Ec, color="tab:blue", label=r"$E_c$"); ax.plot(z, Ev, color="tab:red", label=r"$E_v$")
    ax.axhline(0, color="k", ls="--", lw=0.7, label=r"$E_F$"); ax.axvspan(0, (zc-zb)*1e9, color="tab:orange", alpha=0.18)
    ax.set_xlim(-120, 300); ax.set_xlabel(r"depth $z$ (nm), 0 = top of the sheet"); ax.set_title(lab, fontsize=8)
    ax.grid(alpha=0.3); ax.legend(loc="center right")
axs[0].set_ylabel("energy (eV)")
save(fig, "Fig6")
# Fig 6: three-layer stack, two sheet dopings
s_R1, s_R3 = 200.0, 21.6; d1, ei, d2, rcc, Rmx = 300e-9, 30e-9, 6e-6, 200e-9, 20e-6
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5))
for ax, (ssh, lab, col) in zip(axs, [(1280.0, r"(a) conductive sheet, $N_D = 10^{17}$ cm$^{-3}$", "tab:blue"),
                                    (2.16, r"(b) resistive sheet, $N_D = 10^{14}$ cm$^{-3}$", "tab:red")]):
    P, zs, rs, v, src, b = sc.solve_stack([(d1, s_R1, 6), (ei, ssh, 4), (d2, s_R3, 40)], Rmx, rcc)
    axn = np.where(P[:, 0] < 1e-14)[0]; za = P[axn, 1]; va = v[axn]; o = np.argsort(za); za, va = za[o], va[o]; top = zs[-1]
    zmap = lambda z: 0.30*z/d1 if z <= d1 else (0.30+0.40*(z-d1)/ei if z <= d1+ei else 0.70+0.30*(z-d1-ei)/(top-d1-ei))
    ax.plot(va/1e3, [zmap(z) for z in za], "-o", ms=1.5, color=col)  # V/I, I = 1 A, in kΩ; ax.axhspan(0.30, 0.70, color="tab:orange", alpha=0.18)
    ax.invert_yaxis(); ax.set_yticks([0.15, 0.5, 0.85])
    ax.set_yticklabels(["Region 1\n300 nm", "sheet\n30 nm", "Region 3\n6 µm"])
    ax.set_xlabel(r"$V/I$ on the axis $r = 0$ (kΩ)"); ax.set_title(lab, fontsize=8)
    ax.grid(alpha=0.3, axis="x")
save(fig, "Fig7")

# Fig 5: certified brackets (jce_brackets.json)
BR = json.load(open("jce_brackets.json"))
fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
def draw(a, d, col, lab, dx=1.0):
    nt = [x["ntet"]*dx for x in d]; lo = [x["Rlo"] for x in d]; hi = [x["Rhi"] for x in d]
    a.vlines(nt, lo, hi, color=col, lw=3, alpha=0.55, label=lab)
    a.plot(nt, lo, "_", color=col, ms=8); a.plot(nt, hi, "_", color=col, ms=8)
    for x, n in zip(d, nt): a.text(n*1.04, x["Rhi"], "%.0f%%" % x["eta"], fontsize=6, va="bottom")
draw(ax[0], BR["bilayer"], "tab:purple", r"certified interval $[R_{lo}, R_{up}]$")
ax[0].set_ylabel(r"access resistance $R$ (non-dimensional)"); ax[0].legend(loc="lower left")
draw(ax[1], BR["stack"], "tab:red", r"resistive sheet, $N_D = 10^{14}$ cm$^{-3}$")
draw(ax[1], BR["stack_conductive"], "tab:blue", r"conductive sheet, $N_D = 10^{17}$ cm$^{-3}$", dx=0.93)
ax[1].set_ylabel(r"access resistance $R$ (Ω)"); ax[1].legend(loc="center right", fontsize=6.5)
for k, tag in enumerate(["(a)", "(b)"]):
    ax[k].set_xscale("log"); ax[k].set_xlabel("number of tetrahedra"); ax[k].grid(alpha=0.3, which="both")
    ax[k].text(0.93, 0.92, tag, transform=ax[k].transAxes)
save(fig, "Fig5")
print(json.dumps(sweep, indent=1))
