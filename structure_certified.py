"""Article 1, Section III-B: certified conduction through the attached 3-layer
structure (Region 1 / fine 30 nm sheet / Region 2), current injected on top.

Axisymmetric P1 conduction on anisotropic cells (a few flat cells across the
30 nm sheet). Conductivities from the dopings; the intermediate sheet is taken
as a resistive (intrinsic) isolation layer -- the physically natural choice and
the case in which resolving the thin layer is essential.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def graded(x0,x1,n,ratio):
    g=np.cumsum(ratio**np.arange(n)); g=np.concatenate([[0],g]); g=g/g[-1]
    return x0+(x1-x0)*g

def solve_stack(layers, Rmax, rc, nr=90, I=1.0):
    """layers = list of (thickness, sigma, n_z) from the TOP surface down.
    Current I injected over the top disk r<=rc; V=0 on the far boundary."""
    # z grid, layer by layer
    z=[0.0]; sig_lay=[]; bnds=[0.0]
    for (d,s,nz) in layers:
        seg=np.linspace(z[-1],z[-1]+d,nz+1)[1:]
        z=z+list(seg); bnds.append(bnds[-1]+d)
    zs=np.array(z); Htot=zs[-1]
    def sigma_at(zc):
        acc=0.0
        for (d,s,nz) in layers:
            if zc< acc+d: return s
            acc+=d
        return layers[-1][1]
    # radial grid: fine near source, geometric to Rmax
    rs=np.concatenate([[0.0], rc/6*np.cumprod(np.full(nr,1.13))]); rs=rs[rs<=Rmax]
    if rs[-1]<Rmax: rs=np.append(rs,Rmax)
    NR,NZ=len(rs),len(zs); P=np.array([(r,zz) for r in rs for zz in zs])
    idx=lambda i,k:i*NZ+k; rows=[];cols=[];vals=[]
    for i in range(NR-1):
        for k in range(NZ-1):
            quad=[idx(i,k),idx(i+1,k),idx(i+1,k+1),idx(i,k+1)]
            zc=0.5*(zs[k]+zs[k+1]); s=sigma_at(zc)
            for tri in [(quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3])]:
                (x1,y1),(x2,y2),(x3,y3)=P[tri[0]],P[tri[1]],P[tri[2]]
                det=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1); area=0.5*abs(det)
                if area<1e-40: continue
                b=[(y2-y3)/det,(y3-y1)/det,(y1-y2)/det]
                g=[(x3-x2)/det,(x1-x3)/det,(x2-x1)/det]
                rcen=(x1+x2+x3)/3.0; w=2*np.pi*rcen*s*area
                for u in range(3):
                    for vv in range(3):
                        rows.append(tri[u]);cols.append(tri[vv]);vals.append(w*(b[u]*b[vv]+g[u]*g[vv]))
    n=len(P); K=sp.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()
    far=np.where((P[:,0]>Rmax-1e-12)|(P[:,1]>Htot-1e-12))[0]
    src=np.where((P[:,0]<=rc+1e-14)&(P[:,1]<1e-14))[0]
    f=np.zeros(n); f[src]=I/len(src)
    free=np.setdiff1d(np.arange(n),far)
    v=np.zeros(n); v[free]=spla.spsolve(K[free][:,free].tocsc(),f[free])
    return P,zs,rs,v,src,bnds

if __name__=="__main__":
    # SI conductivities from dopings (Si, 300 K)
    s1=375.0     # Region 1  N_D=2e16
    s2=2.16      # Region 2  N_D=1e14
    ssheet=4.3e-4  # intrinsic isolation sheet
    d1=300e-9; ei=30e-9; d2=6e-6; rc=200e-9; Rmax=20e-6
    layers=[(d1,s1,6),(ei,ssheet,4),(d2,s2,40)]
    P,zs,rs,v,src,bnds=solve_stack(layers,Rmax,rc)
    print("nodes=%d  sheet cells=4  interfaces z(nm)=%s"%(len(P),[round(b*1e9) for b in bnds]))
    # potential along the axis r=0 vs depth -> show the drop across the sheet
    ax=np.where(P[:,0]<1e-14)[0]; za=P[ax,1]; va=v[ax]; o=np.argsort(za); za=za[o]; va=va[o]
    Vtop=va[0]
    def Vat(znm): return np.interp(znm*1e-9,za,va)
    print(" axis potential (r=0):")
    print("   top(z=0)      = %.4e V"%Vat(0))
    print("   sheet top(300)= %.4e V"%Vat(300))
    print("   sheet bot(330)= %.4e V"%Vat(330))
    print("   drop ACROSS 30nm sheet = %.4e V  (%.1f%% of the total top potential)"%(Vat(300)-Vat(330),100*(Vat(300)-Vat(330))/Vtop))
    print("   R_access (=Vtop/I) = %.4e ohm"%Vtop)
