"""Article 1, Section III-A: certified anisotropic solve vs the analytic
surface potential of a point current source on a thin film / substrate.

Analytic (semi-infinite substrate, insulating top except the source):
    V(r) = I/(2 pi s1) [ 1/r + 2 sum_{n>=1} k^n / sqrt(r^2+(2 n h)^2) ],
    k = (s1 - s2)/(s1 + s2),   h = film thickness.

Numerical: axisymmetric P1 conduction on a graded (r,z) mesh -- a coarse radial
grid with a few flat cells across the film -- exactly the anisotropic-cell idea
of the paper. The surface potential is compared to the analytic series; the
stretched film cells must reproduce it.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def analytic_Vsurf(r, h, s1, s2, I=1.0, N=2000):
    k=(s1-s2)/(s1+s2); s=1.0/r
    for n in range(1,N+1): s=s+2*k**n/np.sqrt(r**2+(2*n*h)**2)
    return I/(2*np.pi*s1)*s

def graded(x0,x1,n,ratio):
    if abs(ratio-1)<1e-9: return np.linspace(x0,x1,n+1)
    g=np.cumsum(ratio**np.arange(n)); g=np.concatenate([[0],g]); g=g/g[-1]
    return x0+(x1-x0)*g

def solve_axisym(h,e2,s1,s2,Rmax,a_src, nr=60,nz1=4,nz2=24, I=1.0):
    # radial nodes: fine near source (r ~ a_src), geometric out to Rmax
    rs=np.concatenate([[0.0], a_src*np.cumprod(np.full(nr,1.12))])
    rs=rs[rs<=Rmax]; 
    if rs[-1]<Rmax: rs=np.append(rs,Rmax)
    # z nodes: 0..h film (uniform), h..h+e2 substrate (graded, fine near film)
    zf=np.linspace(0,h,nz1+1)
    zsub=h+ (graded(0,e2,nz2,1.18))[1:]
    zs=np.concatenate([zf,zsub])
    NR,NZ=len(rs),len(zs); pts=np.array([(r,z) for r in rs for z in zs])
    idx=lambda i,k:i*NZ+k
    tris=[]; sig=[]
    for i in range(NR-1):
        for k in range(NZ-1):
            a=idx(i,k);b=idx(i+1,k);c=idx(i,k+1);d=idx(i+1,k+1)
            zc=0.5*(zs[k]+zs[k+1]); s=s1 if zc<h else s2
            tris.append((a,b,d)); sig.append(s)
            tris.append((a,d,c)); sig.append(s)
    tris=np.array(tris); sig=np.array(sig); P=pts
    # axisymmetric P1 stiffness: Ke = 2 pi r_c sigma area (grad.grad)
    rows=[];cols=[];vals=[]
    for t,s in zip(tris,sig):
        (x1,y1),(x2,y2),(x3,y3)=P[t[0]],P[t[1]],P[t[2]]
        det=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1); area=0.5*abs(det)
        if area<1e-30: continue
        b=[(y2-y3)/det,(y3-y1)/det,(y1-y2)/det]
        g=[(x3-x2)/det,(x1-x3)/det,(x2-x1)/det]
        rc=(x1+x2+x3)/3.0
        w=2*np.pi*rc*s*area
        for u in range(3):
            for v in range(3):
                rows.append(t[u]);cols.append(t[v]);vals.append(w*(b[u]*b[v]+g[u]*g[v]))
    n=len(P); K=sp.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()
    # BC: V=0 on far boundary (r=Rmax or z=h+e2); source current I at top-centre nodes r<=a_src,z=0
    far=np.where((P[:,0]>Rmax-1e-9)|(P[:,1]>zs[-1]-1e-9))[0]
    f=np.zeros(n)
    src=np.where((P[:,0]<=a_src+1e-12)&(P[:,1]<1e-12))[0]
    f[src]=I/len(src)
    free=np.setdiff1d(np.arange(n),far)
    Kff=K[free][:,free].tocsc()
    v=np.zeros(n); v[free]=spla.spsolve(Kff,f[free])
    return P,zs,rs,v,src

if __name__=="__main__":
    h=30e-9; e2=6e-6; s1=100.0; s2=1.0; Rmax=8e-6; a=h/2
    P,zs,rs,v,src=solve_axisym(h,e2,s1,s2,Rmax,a)
    # surface nodes z=0
    surf=np.where(P[:,1]<1e-12)[0]; rsurf=P[surf,0]; vsurf=v[surf]
    o=np.argsort(rsurf); rsurf=rsurf[o]; vsurf=vsurf[o]
    print("aspect film cell dr/dz near source ~ %.0f"%((rs[2]-rs[1])/(h/4)))
    print(" r/h    V_num        V_analytic    ratio")
    for rr in [1,2,4,8,20,50]:
        rt=rr*h
        if rt>rsurf[-1]: continue
        vn=np.interp(rt,rsurf,vsurf); va=analytic_Vsurf(rt,h,s1,s2)
        print("  %5.0f  %.4e  %.4e  %.4f"%(rr,vn,va,vn/va))
