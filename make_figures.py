"""Figures 1-9 of the article, at print size (84 mm single column, 174 mm double column).
Vector PDF + EPS in figures/. Reads jce_step.json, jce_contact.json, jce_brackets.json;
recomputes the aspect-ratio sweep (jce_aspect.json), the band diagrams and the axis profiles."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import bilayer_certified as bc
import structure_certified as sc
import band_diagram as bd
import tlm_contact as tc

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42})
os.makedirs("figures", exist_ok=True)
MM = 1 / 25.4
COL1, COL2 = 84 * MM, 174 * MM


def save(fig, name):
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "eps"):
        fig.savefig(f"figures/{name}.{ext}")
    plt.close(fig)


# ---- Fig 1: thin layer in a large body
fig, ax = plt.subplots(figsize=(COL2, 1.55))
ax.add_patch(Rectangle((0, 0), 10, 1.6, fc="#d8dee6", ec="k", lw=0.6))
ax.add_patch(Rectangle((0, 1.6), 10, 0.06, fc="#2f8f7f", ec="none"))
ax.annotate("", xy=(10.3, 1.57), xytext=(10.3, 1.69), arrowprops=dict(arrowstyle="<->", color="#c0601b", lw=0.7))
ax.text(10.5, 1.63, "layer thickness: 1 to 100 nm", color="#c0601b", va="center")
ax.annotate("", xy=(0, -0.3), xytext=(10, -0.3), arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(5, -0.62, "lateral size: 1 µm to 10 mm", ha="center")
ax.text(5, 0.8, "body / substrate", ha="center", color="#444")
ax.text(5, 1.95, "thin layer", ha="center", color="#2f8f7f")
ax.set_xlim(-0.3, 14.3); ax.set_ylim(-0.9, 2.3); ax.axis("off")
save(fig, "Fig1")

# ---- Fig 2: hand example, step conductor
S = json.load(open("jce_step.json"))
fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
shape = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
for ax, mode in zip(axs[:2], ("slices", "tubes")):
    ax.add_patch(Polygon(shape, closed=True, fc="#eef1f5", ec="k", lw=0.7))
    ax.plot([0, 0], [0, 2], color="tab:red", lw=2.5); ax.plot([2, 2], [0, 1], color="tab:blue", lw=2.5)
    if mode == "slices":
        for x in np.arange(0.2, 1.0, 0.2): ax.plot([x, x], [0, 2], color="#888", lw=0.5)
        for x in np.arange(1.0, 2.0, 0.2): ax.plot([x, x], [0, 1], color="#888", lw=0.5)
        ax.text(1, -0.25, "(a) equipotential slices\n" + r"$R \geq 1/2 + 1 = 1.5$", ha="center", va="top", fontsize=7.5)
    else:
        ax.add_patch(Rectangle((0, 1), 1, 1, fc="#cfd4da", ec="none"))
        for y in np.arange(0.2, 1.0, 0.2):
            ax.annotate("", xy=(2, y), xytext=(0, y), arrowprops=dict(arrowstyle="->", color="#888", lw=0.5))
        ax.text(0.5, 1.5, "no current", ha="center", va="center", fontsize=6.5, color="#555")
        ax.text(1, -0.25, "(b) straight current tubes\n" + r"$R \leq 1 + 1 = 2$", ha="center", va="top", fontsize=7.5)
    ax.text(-0.08, 1.0, "$V=1$", color="tab:red", rotation=90, va="center", ha="right", fontsize=7)
    ax.text(2.08, 0.5, "$V=0$", color="tab:blue", rotation=90, va="center", ha="left", fontsize=7)
    ax.set_xlim(-0.35, 2.35); ax.set_ylim(-0.95, 2.1); ax.set_aspect("equal"); ax.axis("off")
ax = axs[2]
nn = [d["nodes"] for d in S]
ax.vlines(nn, [d["R_lo"] for d in S], [d["R_up"] for d in S], color="tab:purple", lw=3, alpha=0.55)
ax.plot(nn, [d["R_lo"] for d in S], "_", color="tab:purple", ms=7); ax.plot(nn, [d["R_up"] for d in S], "_", color="tab:purple", ms=7)
ax.set_xscale("log"); ax.set_xlabel("number of nodes"); ax.set_ylabel(r"$[R_{lo},\,R_{up}]$ (units of $1/\sigma$)")
ax.text(0.95, 0.92, "(c)", transform=ax.transAxes, ha="right"); ax.grid(alpha=0.3, which="both")
save(fig, "Fig2")

# ---- Fig 3: isotropic vs anisotropic discretization
fig, axs = plt.subplots(1, 2, figsize=(COL2, 1.75))
for ax, mode in zip(axs, ["iso", "aniso"]):
    ax.add_patch(Rectangle((0, 0), 10, 3, fc="#eef1f5", ec="k", lw=0.6))
    ax.add_patch(Rectangle((0, 2.6), 10, 0.4, fc="#d7f0ea", ec="none"))
    if mode == "iso":
        for x in np.arange(0, 10.01, 0.4): ax.plot([x, x], [0, 3], color="#c33", lw=0.25)
        for y in np.arange(0, 3.01, 0.4): ax.plot([0, 10], [y, y], color="#c33", lw=0.25)
        ax.text(5, -0.45, "(a) near-square cells: in-plane step = layer thickness", ha="center")
    else:
        for x in np.arange(0, 10.01, 1.25): ax.plot([x, x], [0, 3], color="#2a7", lw=0.45)
        for y in [0, 0.7, 1.4, 2.1, 2.6, 2.73, 2.87, 3.0]: ax.plot([0, 10], [y, y], color="#2a7", lw=0.45)
        ax.text(5, -0.45, "(b) flat cells in the layer, coarse lateral grid", ha="center")
    ax.text(5, 2.75, "layer", ha="center", color="#2f8f7f", fontsize=7)
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.8, 3.2); ax.axis("off")
save(fig, "Fig3")

# ---- Fig 4: aspect-ratio sweep, layer-parallel flow (exact G known)
W, e1, e2, s1, s2 = 1.0, 0.002, 0.10, 5.0, 1.0
sweep = []
for nx, nz1, nz2 in [(50, 1, 2), (25, 1, 2), (10, 1, 2), (5, 1, 4), (4, 2, 4), (8, 2, 4), (4, 4, 6), (2, 4, 6)]:
    o = bc.certified_bilayer(W, e1, e2, s1, s2, contacts="lateral", nx=nx, ny=nx, nz1=nz1, nz2=nz2)
    rex = 1 / o["g_exact"]; rlo, rhi = o["r"]
    sweep.append(dict(nx=nx, nz1=nz1, nz2=nz2, ntet=int(o["ntet"]), aspect=float(o["aspect"]),
                      gap=float(abs(rhi - rlo) / rhi), inside=bool(rlo <= rex * (1 + 1e-12) and rex <= rhi * (1 + 1e-12)),
                      relerr_up=float(abs(rlo - rex) / rex)))
sweep.sort(key=lambda d: d["aspect"]); json.dump(sweep, open("jce_aspect.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(COL1, 2.3))
asp = [d["aspect"] for d in sweep]
ax.loglog(asp, [max(d["gap"], 1e-16) for d in sweep], "o-", color="#2a7", ms=3.5, label=r"relative width $\eta$")
ax.axhline(2.2e-16, color="k", ls=":", lw=0.7); ax.text(asp[0], 4e-16, "double-precision epsilon", fontsize=6.5)
ax.set_xlabel("cell aspect ratio in the layer"); ax.set_ylabel(r"relative width $\eta$")
ax.set_ylim(1e-17, 1e-2); ax.grid(alpha=0.3, which="major"); ax.legend(loc="upper right")
save(fig, "Fig4")

# ---- Fig 5: contact on a thin film: structure and current lines near the contact edge
C = json.load(open("jce_contact.json")); p = C["params"]
t, Rsh, delta, L, g = p["t"], p["Rsh"], p["delta"], p["L"], p["g"]
fig = plt.figure(figsize=(COL2, 2.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])
ax = fig.add_subplot(gs[:, 0])
ax.add_patch(Rectangle((0, 0), 10, 1.2, fc="#d8dee6", ec="k", lw=0.6))                 # film
ax.add_patch(Rectangle((0, 1.2), 4, 0.22, fc="#f2c58c", ec="k", lw=0.4))               # interfacial layer
ax.add_patch(Rectangle((0, 1.42), 4, 0.7, fc="#777", ec="k", lw=0.4))                  # metal
ax.text(2, 1.77, "metal, $V = V_0$", ha="center", va="center", color="w", fontsize=7)
ax.text(5.6, 1.62, r"interfacial layer $\delta$, $\rho_c$", fontsize=7, color="#a05a00", va="center")
ax.annotate("", xy=(4.05, 1.31), xytext=(5.5, 1.6), arrowprops=dict(arrowstyle="-", color="#a05a00", lw=0.5))
ax.text(5, 0.6, r"film: thickness $t$, sheet resistance $R_{sh}$", ha="center", va="center", fontsize=7)
ax.plot([10, 10], [0, 1.2], color="tab:blue", lw=2.5); ax.text(10.15, 0.6, "$V=0$\n(mid-gap)", color="tab:blue", fontsize=6.5, va="center")
for (x0, x1, lab) in [(0, 4, "contact length $L$"), (4, 10, "half gap $g$")]:
    ax.annotate("", xy=(x0, -0.35), xytext=(x1, -0.35), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(0.5 * (x0 + x1), -0.75, lab, ha="center", fontsize=7)
ax.text(5, 2.75, "(a) half structure (not to scale)", ha="center")
ax.set_xlim(-0.3, 12.6); ax.set_ylim(-1.1, 3.1); ax.axis("off")
for k, (rc_cm2, tag) in enumerate([(1e-6, "(b)"), (1e-9, "(c)")]):
    ax = fig.add_subplot(gs[k, 1])
    r = tc.solve(t, Rsh, rc_cm2 * 1e-4, delta, L, g, 5, "graded")
    nodes, tris, V, psi, xs, zs = r["mesh"]
    x = (nodes[:, 0] - L) * 1e9; z = nodes[:, 1] * 1e9
    ax.tricontour(x, z, tris, psi, levels=np.linspace(0.05, 0.95, 10), colors="tab:blue", linewidths=0.6)
    ax.add_patch(Rectangle((-400, t * 1e9 + delta * 1e9), 400, 12, fc="#777", ec="none"))
    ax.set_xlim(-400, 150); ax.set_ylim(0, t * 1e9 + 12)
    LT = 1e9 * np.sqrt(rc_cm2 * 1e-4 / Rsh)
    ax.set_title(r"%s $\rho_c = 10^{%d}$ $\Omega$ cm$^2$, $L_T$ = %.0f nm" % (tag, int(np.log10(rc_cm2)), LT), fontsize=7.5)
    ax.set_ylabel("$z$ (nm)")
    if k == 1: ax.set_xlabel("distance from the contact edge (nm)")
save(fig, "Fig5")

# ---- Fig 6: contact resistance: intervals for the two mesh families; deviation of the 1D formula
fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
for fam, col, lab, dx in [("graded", "tab:purple", "sequence G (graded everywhere)", 1.0), ("edge", "tab:orange", "sequence E (contact edge only)", 1.0)]:
    d = C[fam]; nn = [q["nodes"] * dx for q in d]
    ax[0].vlines(nn, [q["Rc_lo"] for q in d], [q["Rc_up"] for q in d], color=col, lw=3, alpha=0.6, label=lab)
    ax[0].plot(nn, [q["Rc_lo"] for q in d], "_", color=col, ms=7); ax[0].plot(nn, [q["Rc_up"] for q in d], "_", color=col, ms=7)
ax[0].axhline(10.0, color="k", ls="--", lw=0.7); ax[0].text(2.2e4, 10.02, "1D formula", fontsize=6.5)
ax[0].set_xscale("log"); ax[0].set_xlabel("number of nodes"); ax[0].set_ylabel(r"$[R_{lo},\,R_{up}]$ of $R_c W$ ($\Omega$ µm)")
ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3, which="both"); ax[0].text(0.03, 0.12, "(a)", transform=ax[0].transAxes)
cv = C["curve"]; xx = [q["t_over_LT"] for q in cv]
ax[1].loglog(xx, [q["dev_lo"] for q in cv], "o-", ms=3, color="tab:green", label=r"$R_c$: guaranteed minimum excess over the 1D formula")
ax[1].loglog(xx, [100 * (q["app_lo"] - 1) for q in cv], "s-", ms=3, color="tab:red", label=r"$\rho_c$: guaranteed minimum overestimate by 1D extraction")
ax[1].set_xlabel(r"film thickness / transfer length, $t/L_T$"); ax[1].set_ylabel("deviation (%)")
ax[1].legend(loc="upper left", fontsize=6.2); ax[1].grid(alpha=0.3, which="both"); ax[1].text(0.93, 0.05, "(b)", transform=ax[1].transAxes)
save(fig, "Fig6")

# ---- Fig 7: equilibrium band diagrams of the stack
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
for ax, (Nsh, lab) in zip(axs, [(1e17, r"(a) sheet $N_D = 10^{17}$ cm$^{-3}$"), (1e14, r"(b) sheet $N_D = 10^{14}$ cm$^{-3}$")]):
    zs, phi, (zb, zc) = bd.solve_band(1e16, Nsh, 1e15)
    Ec = bd.EC0 - phi; Ev = Ec - bd.EG; z = (zs - zb) * 1e9
    ax.plot(z, Ec, color="tab:blue", label=r"$E_c$"); ax.plot(z, Ev, color="tab:red", label=r"$E_v$")
    ax.axhline(0, color="k", ls="--", lw=0.7, label=r"$E_F$"); ax.axvspan(0, (zc - zb) * 1e9, color="tab:orange", alpha=0.18)
    ax.set_xlim(-120, 300); ax.set_xlabel(r"depth $z$ (nm), 0 = top of the sheet"); ax.set_title(lab, fontsize=8)
    ax.grid(alpha=0.3); ax.legend(loc="center right")
axs[0].set_ylabel("energy (eV)")
save(fig, "Fig7")

# ---- Fig 8: three-layer stack, potential on the axis for the two sheet dopings
s_R1, s_R3 = 200.0, 21.6; d1, ei, d2, rcc, Rmx = 300e-9, 30e-9, 6e-6, 200e-9, 20e-6
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5))
for ax, (ssh, lab, col) in zip(axs, [(1280.0, r"(a) heavily doped sheet, $N_D = 10^{17}$ cm$^{-3}$", "tab:blue"),
                                    (2.16, r"(b) lightly doped sheet, $N_D = 10^{14}$ cm$^{-3}$", "tab:red")]):
    P, zs, rs, v, src, b = sc.solve_stack([(d1, s_R1, 6), (ei, ssh, 4), (d2, s_R3, 40)], Rmx, rcc)
    axn = np.where(P[:, 0] < 1e-14)[0]; za = P[axn, 1]; va = v[axn]; o = np.argsort(za); za, va = za[o], va[o]; top = zs[-1]
    zmap = lambda z: 0.30 * z / d1 if z <= d1 else (0.30 + 0.40 * (z - d1) / ei if z <= d1 + ei else 0.70 + 0.30 * (z - d1 - ei) / (top - d1 - ei))
    ax.plot(va / 1e3, [zmap(z) for z in za], "-o", ms=1.5, color=col)
    ax.axhspan(0.30, 0.70, color="tab:orange", alpha=0.18)
    ax.invert_yaxis(); ax.set_yticks([0.15, 0.5, 0.85])
    ax.set_yticklabels(["Region 1\n300 nm", "sheet\n30 nm", "Region 3\n6 µm"])
    ax.set_xlabel(r"$V/I$ on the axis $r = 0$ (k$\Omega$)"); ax.set_title(lab, fontsize=8)
    ax.grid(alpha=0.3, axis="x")
save(fig, "Fig8")

# ---- Fig 9: intervals of the access resistance of the stack, 3D
BR = json.load(open("jce_brackets.json"))
fig, ax = plt.subplots(figsize=(COL1, 2.5))
for d, col, lab, dx in [(BR["stack"], "tab:red", r"lightly doped sheet, $N_D = 10^{14}$ cm$^{-3}$", 1.0),
                        (BR["stack_conductive"], "tab:blue", r"heavily doped sheet, $N_D = 10^{17}$ cm$^{-3}$", 0.93)]:
    nt = [q["ntet"] * dx for q in d]; lo = [q["Rlo"] for q in d]; hi = [q["Rhi"] for q in d]
    ax.vlines(nt, lo, hi, color=col, lw=3, alpha=0.55, label=lab)
    ax.plot(nt, lo, "_", color=col, ms=8); ax.plot(nt, hi, "_", color=col, ms=8)
    for q, n in zip(d, nt): ax.text(n * 1.05, q["Rhi"], "%.0f%%" % q["eta"], fontsize=6, va="bottom")
ax.set_xscale("log"); ax.set_xlabel("number of tetrahedra"); ax.set_ylabel(r"access resistance $R$ ($\Omega$)")
ax.set_xlim(5e3, 1.3e5); ax.set_ylim(850, 2200); ax.legend(loc="upper right", fontsize=6.5); ax.grid(alpha=0.3, which="both")
save(fig, "Fig9")
print("figures written")
