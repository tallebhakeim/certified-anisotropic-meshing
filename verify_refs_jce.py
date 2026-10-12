"""Crossref check of the references ADDED for the JCE version: prints the
canonical metadata of the best match for each title, to compare by hand."""
import json, sys, time, urllib.parse, urllib.request
UA = "refcheck/1.0 (mailto:hakeim.talleb@sorbonne-universite.fr)"
Q = [
 "An a posteriori residual error estimator for the finite element method on anisotropic tetrahedral meshes",
 "Edge residuals dominate a posteriori error estimates for linear finite element methods on anisotropic triangular and tetrahedral meshes",
 "On the angle condition in the finite element method",
 "On the maximum angle condition for linear tetrahedral elements",
 "Anisotropic mesh adaptation for CFD computations",
 "Continuous mesh framework part I: well-posed continuous interpolation error",
 "An anisotropic error indicator based on Zienkiewicz-Zhu error estimator: application to elliptic and parabolic problems",
 "Guaranteed and fully robust a posteriori error estimates for conforming discretizations of diffusion problems with discontinuous coefficients",
 "Polynomial-degree-robust a posteriori estimates in a unified setting for conforming, nonconforming, discontinuous Galerkin, and mixed discretizations",
 "Equilibrated residual error estimates are p-robust",
 "Explicit error bounds in a conforming finite element method",
 "A posteriori error estimation in finite element analysis Ainsworth Oden",
 "Mixed finite element methods and applications Boffi Brezzi Fortin",
 "Large-signal analysis of a silicon Read diode oscillator",
 "Numerical methods for semiconductor device simulation Bank Rose Fichtner",
 "On the interplay between meshing and discretization in three-dimensional diffusion simulation",
 "Moving mesh adaptation for Si and GaN-based power device simulation",
 "TCAD process/device modeling challenges and opportunities for the next decade",
 "Measurement of sheet resistivities with the four-point probe",
 "Semiconductor material and device characterization Schroder",
 "Single-layer MoS2 transistors",
 "Oxide semiconductor thin-film transistors: a review of recent advances",
 "FinFETs and other multi-gate transistors",
 "A posteriori estimates for partial differential equations Repin",
 "A method for measuring specific resistivity and Hall effect of discs of arbitrary shape",
 "Nano-electronic Simulation Software (NESS): a flexible nano-device simulation platform",
]
for q in Q:
    u = "https://api.crossref.org/works?rows=2&query.bibliographic=" + urllib.parse.quote(q)
    try:
        it = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=30))["message"]["items"]
    except Exception as e:
        print("ERR", q, e); continue
    print("\n?", q)
    for x in it[:2]:
        au = ", ".join(a.get("family", "?") for a in x.get("author", [])[:4])
        print("  ->", x.get("issued", {}).get("date-parts", [[None]])[0][0], "|", au, "|", (x.get("title") or ["?"])[0][:120], "|",
              (x.get("container-title") or [""])[0][:50], x.get("volume"), x.get("issue"), x.get("page"), "|", x["DOI"])
    time.sleep(0.3)
