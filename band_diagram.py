"""Equilibrium energy-band diagram of the attached stack (Region1 n+ / 30 nm
sheet / Region2 n-), from the certified Poisson-Boltzmann equilibrium.

NB: a band diagram is an EQUILIBRIUM quantity (nonlinear Poisson-Boltzmann),
i.e. the physics moved to the companion paper; Article 1 itself is linear
conduction. This produces it so the option is on the table.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

Q=1.602176634e-19; KT=0.025852; NI=1e16      # m^-3 (1e10 cm^-3)
EPS=11.7*8.8541878128e-12; CM3=1e6
EG=1.12; NC=2.8e25; EC0=KT*np.log(NC/NI)      # eV, E_c - intrinsic level

def solve_band(N1_cm3, Nsheet_cm3, N2_cm3, t_sheet=30e-9, L1=400e-9, L2=6e-6,
               n_sheet=30, n_out=260):
    """1-D equilibrium: -lam^2 u'' + sinh(u) = g(z), g = N_net/(2 NI).
    Region1 [0,L1] N1, sheet [L1,L1+t] Nsheet, Region2 [L1+t, L1+t+L2] N2."""
    N1,Ns,N2 = N1_cm3*CM3, Nsheet_cm3*CM3, N2_cm3*CM3
    lam2 = EPS*KT/(2*Q*NI)
    za,zb,zc,zd = 0.0, L1, L1+t_sheet, L1+t_sheet+L2
    # graded grid: fine inside/around the sheet
    zs=np.concatenate([np.linspace(zb,zc,n_sheet+1),
                       zb-np.geomspace(t_sheet/n_sheet, L1, n_out)[::-1],
                       zc+np.geomspace(t_sheet/n_sheet, L2, n_out)])
    zs=np.unique(np.clip(zs,za,zd))
    def Nnet(z): return N1 if z<zb else (Ns if z<zc else N2)
    g=np.array([Nnet(0.5*(zs[i]+zs[i+1]) if i<len(zs)-1 else zs[i])/(2*NI) for i in range(len(zs))])
    u=np.array([np.arcsinh(Nnet(z)/(2*NI)) for z in zs])   # start at local neutral
    ubL, ubR = np.arcsinh(N1/(2*NI)), np.arcsinh(N2/(2*NI))
    u[0],u[-1]=ubL,ubR
    h=np.diff(zs); idx=np.arange(1,len(zs)-1)
    for it in range(200):
        fl=(u[idx]-u[idx-1])/h[idx-1]; fr=(u[idx+1]-u[idx])/h[idx]
        m=0.5*(h[idx-1]+h[idx])
        F=-lam2*(fr-fl)+m*(np.sinh(u[idx])-g[idx])
        dl=-lam2/h[idx-1]; du=-lam2/h[idx]; dc=lam2*(1/h[idx-1]+1/h[idx])+m*np.cosh(u[idx])
        # Thomas
        cp=np.empty(len(idx)); dp=np.empty(len(idx)); cp[0]=du[0]/dc[0]; dp[0]=F[0]/dc[0]
        for k in range(1,len(idx)):
            den=dc[k]-dl[k]*cp[k-1]; cp[k]=du[k]/den; dp[k]=(F[k]-dl[k]*dp[k-1])/den
        dx=np.empty(len(idx)); dx[-1]=dp[-1]
        for k in range(len(idx)-2,-1,-1): dx[k]=dp[k]-cp[k]*dx[k+1]
        u[idx]-=np.clip(dx,-2,2)
        if np.max(np.abs(dx))<1e-11: break
    phi=KT*u
    return zs, phi, (zb,zc)

if __name__=="__main__":
    # Two cases (per M. Ben Chouikha): R1=1e16, R3=1e15, sheet 1e17 or 1e14
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    cases=[(1e17,'Case 1: sheet N_D = 1e17 (heavily doped -> conductive notch)'),
           (1e14,'Case 2: sheet N_D = 1e14 (lightly doped -> resistive barrier)')]
    for ax,(Nsh,title) in zip(axs,cases):
        zs,phi,(zb,zc)=solve_band(1e16, Nsh, 1e15)
        Ec=EC0-phi; Ev=Ec-EG; z_nm=(zs-zb)*1e9
        ax.plot(z_nm,Ec,color='tab:blue',lw=2,label='E_c')
        ax.plot(z_nm,Ev,color='tab:red',lw=2,label='E_v')
        ax.axhline(0,color='k',ls='--',lw=1.2,label='E_F')
        ax.axvspan(0,(zc-zb)*1e9,color='tab:orange',alpha=0.15)
        ax.text((zc-zb)*1e9/2,Ec.max()+0.015,'30 nm sheet',ha='center',fontsize=8,color='#b60')
        ax.set_xlim(-120,300); ax.set_xlabel('depth z [nm] (0 = top of sheet)')
        ax.set_ylabel('energy [eV]'); ax.legend(fontsize=8,loc='center right')
        ax.set_title(title,fontsize=9.5); ax.grid(alpha=0.3)
    fig.suptitle('Equilibrium band diagram: Region 1 (N_D=1e16) / 30 nm sheet / '
                 'Region 3 (N_D=1e15), two sheet dopings',fontsize=10)
    fig.tight_layout(); fig.savefig('Fig_band_diagram.png',dpi=300)
    print('saved Fig_band_diagram.png (two cases)')
