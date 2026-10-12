"""Certified brackets of Sections 3.1 and 3.2 with a SQUARE top contact
[0.3W, 0.7W]^2 whose edges lie on grid lines for every mesh (nx multiple of 10).
The continuous problem is then the same on every mesh, so every interval
encloses the same exact resistance and their intersection is also certified.
(certified_bracket.py used a node-sampled disk, which changes with the mesh.)"""
import sys, time, numpy as np
import certified_bracket as cb
from bilayer_certified import bilayer_mesh
from conductance3d import conductance_upper_3d, conductance_lower_3d

def _contacts(P, W, top):
    tol = 1e-9*W
    A = np.where((np.abs(P[:, 2]-top) < tol) & (P[:, 0] > 0.3*W-tol) & (P[:, 0] < 0.7*W+tol)
                 & (P[:, 1] > 0.3*W-tol) & (P[:, 1] < 0.7*W+tol))[0]
    B = np.where((P[:, 2] < tol) | (P[:, 0] < tol) | (P[:, 0] > W-tol) | (P[:, 1] < tol) | (P[:, 1] > W-tol))[0]
    return A, B

def _bracket(mesh, sig, A, B):
    gu = conductance_upper_3d(mesh, sig, [A, B], 0, 1); gl = conductance_lower_3d(mesh, sig, [A, B], 0, 1)
    return dict(ntet=int(mesh.nt), nnode=int(len(mesh.points)), Rlo=float(1/gu), Rhi=float(1/gl), eta=float(100*(1/gl-1/gu)*gl))

def bilayer(nx, nz1, nz2, W=1.0, e1=0.03, e2=0.5, s1=100.0, s2=1.0):
    mesh, layer = bilayer_mesh(W, e1, e2, nx, nx, nz1, nz2)
    sig = np.where(layer == 1, s1, s2).astype(float)
    A, B = _contacts(mesh.points, W, e1+e2); return _bracket(mesh, sig, A, B)

def stack(nx, ns, W=10e-6, layers=None):
    layers = layers or [(6e-6, 21.6, 6), (30e-9, 2.16, ns), (300e-9, 200.0, 4)]
    mesh, sig, top = cb.stack_mesh(W, layers, nx, nx)
    A, B = _contacts(mesh.points, W, top); return _bracket(mesh, sig, A, B)

if __name__ == "__main__":
    import json
    quick = "--quick" in sys.argv          # skips the 144000-tetrahedron mesh (about 15 min)
    out = dict(bilayer=[], stack=[], stack_conductive=[])
    for args in [(10, 2, 6), (20, 2, 8), (30, 3, 10)] + ([] if quick else [(40, 3, 12)]):
        t = time.time(); o = bilayer(*args); o["nx"] = args[0]; out["bilayer"].append(o)
        print("problem 3", args, o, "%.0fs" % (time.time()-t), flush=True)
    for args in [(10, 2), (20, 3), (30, 3)]:
        t = time.time(); o = stack(*args); o["nx"] = args[0]; out["stack"].append(o)
        print("problem 4", args, o, "%.0fs" % (time.time()-t), flush=True)
    for args in [(10, 2), (20, 3), (30, 3)]:
        t = time.time(); o = stack(*args, layers=[(6e-6, 21.6, 6), (30e-9, 1280.0, args[1]), (300e-9, 200.0, 4)])
        o["nx"] = args[0]; out["stack_conductive"].append(o)
        print("problem 4, conductive sheet", args, o, "%.0fs" % (time.time()-t), flush=True)
    json.dump(out, open("jce_brackets.json", "w"), indent=1)
