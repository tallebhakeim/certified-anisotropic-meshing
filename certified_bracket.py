"""Certified dual bracket (primal P1 upper G + RT0 lower G) for the spreading
resistance of a top contact on a thin-film / substrate stack -- the geometry of
Sections III-A/B -- on anisotropic cells. Reuses the validated 3D dual pair.
"""
import sys, numpy as np
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from bilayer_certified import bilayer_mesh
from conductance3d import conductance_upper_3d, conductance_lower_3d

def bracket_bilayer(W,e1,e2,s1,s2, nx=10,ny=10,nz1=2,nz2=6, cfrac=0.18):
    mesh,layer = bilayer_mesh(W,e1,e2,nx,ny,nz1,nz2)
    sig = np.where(layer==1,s1,s2).astype(float)
    P=mesh.points; top=e1+e2; tol=1e-9*W
    xc=yc=W/2; rc=cfrac*W
    # contact A: top-centre disk ; contact B: far return = bottom + 4 lateral walls
    A=np.where((np.abs(P[:,2]-top)<tol)&((P[:,0]-xc)**2+(P[:,1]-yc)**2<=rc*rc+tol))[0]
    far=(P[:,2]<tol)|(P[:,0]<tol)|(P[:,0]>W-tol)|(P[:,1]<tol)|(P[:,1]>W-tol)
    B=np.where(far)[0]
    gu=conductance_upper_3d(mesh,sig,[A,B],0,1)
    gl=conductance_lower_3d(mesh,sig,[A,B],0,1)
    Rlo,Rhi=1/gu,1/gl
    eta=(Rhi-Rlo)/Rhi
    return dict(ntet=mesh.nt,Glo=gl,Gup=gu,Rlo=Rlo,Rhi=Rhi,eta=eta)

if __name__=="__main__":
    print("III-A  film(sigma1)/substrate(sigma2), top disk contact, far return")
    print("  certified bracket on the access resistance:")
    for (nx,nz1,nz2) in [(8,2,5),(12,2,7),(16,3,9)]:
        o=bracket_bilayer(1.0,0.03,0.5, 100.0,1.0, nx=nx,ny=nx,nz1=nz1,nz2=nz2)
        print("   ntet=%6d  R in [%.5f, %.5f]  certified width eta=%.2f%%"%(
            o['ntet'],o['Rlo'],o['Rhi'],100*o['eta']))

# ---- 3-layer stack (Section III-B) ----
import sys as _sys
from dgm.mesh3d import TetMesh
_KUHN=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
def stack_mesh(W, layers, nx, ny):
    # layers = [(thickness, sigma, nz)] from BOTTOM up
    zs=[0.0]; segs=[]
    for (d,s,nz) in layers:
        for k in range(1,nz+1): zs.append(zs[-1] if False else zs[-1])
    # build z nodes explicitly
    znodes=[0.0]; 
    for (d,s,nz) in layers:
        base=znodes[-1]
        for k in range(1,nz+1): znodes.append(base+d*k/nz)
    zs=np.array(znodes); nz=len(zs)-1
    xs=np.linspace(0,W,nx+1); ys=np.linspace(0,W,ny+1)
    NX,NY,NZ=len(xs),len(ys),len(zs); pts=np.zeros((NX*NY*NZ,3))
    idx=lambda i,j,k:(i*NY+j)*NZ+k
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                n=idx(i,j,k); pts[n]=[xs[i],ys[j],zs[k]]
    def sig_of(zc):
        acc=0.0
        for (d,s,nz2) in layers:
            if zc<acc+d: return s
            acc+=d
        return layers[-1][1]
    tets=[]; sigt=[]
    for i in range(NX-1):
        for j in range(NY-1):
            for k in range(NZ-1):
                c=[idx(i,j,k),idx(i+1,j,k),idx(i,j+1,k),idx(i+1,j+1,k),
                   idx(i,j,k+1),idx(i+1,j,k+1),idx(i,j+1,k+1),idx(i+1,j+1,k+1)]
                c=np.array(c); zc=0.5*(zs[k]+zs[k+1]); s=sig_of(zc)
                for t in _KUHN: tets.append(c[t]); sigt.append(s)
    mesh=TetMesh(pts,np.array(tets))
    zc=mesh.points[mesh.tets][:,:,2].mean(axis=1)
    sig=np.array([sig_of(z) for z in zc])
    return mesh,sig,zs[-1]

def bracket_stack(W, layers, nx=10,ny=10, cfrac=0.18):
    mesh,sig,top=stack_mesh(W,layers,nx,ny); P=mesh.points; tol=1e-9*W
    xc=yc=W/2; rc=cfrac*W
    A=np.where((np.abs(P[:,2]-top)<tol)&((P[:,0]-xc)**2+(P[:,1]-yc)**2<=rc*rc+tol))[0]
    far=(P[:,2]<tol)|(P[:,0]<tol)|(P[:,0]>W-tol)|(P[:,1]<tol)|(P[:,1]>W-tol)
    B=np.where(far)[0]
    gu=conductance_upper_3d(mesh,sig,[A,B],0,1); gl=conductance_lower_3d(mesh,sig,[A,B],0,1)
    return dict(ntet=mesh.nt,Rlo=1/gu,Rhi=1/gl,eta=(1/gl-1/gu)/(1/gl))

if __name__=="__main__":
    print("\nIII-B  3-layer stack: Region1(sigma1)/30nm sheet(intrinsic)/Region2(sigma2)")
    # W=10um lateral; d1=300nm Region1; sheet 30nm; d2=6um Region2 (SI conductivities)
    for (nx,ns) in [(10,2),(14,3)]:
        layers=[(6e-6,2.16,6),(30e-9,4.3e-4,ns),(300e-9,375.0,4)]  # bottom->top
        o=bracket_stack(10e-6,layers,nx=nx,ny=nx)
        print("   ntet=%6d  R in [%.4g, %.4g]  certified width eta=%.2f%%"%(
            o['ntet'],o['Rlo'],o['Rhi'],100*o['eta']))
