#!/usr/bin/env python3
"""Quick smoke test of down2L central-value computations."""
import sys, os, numpy as np
from scipy.integrate import solve_ivp

# Setup paths like down2L.py does
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'downtype-drift' else SCRIPT_DIR
sys.path.insert(0, os.path.join(REPO_ROOT, "trunc-differencing"))

# Import helpers from sm_rge
from sm_rge import ONE_LOOP_FACTOR, TWO_LOOP_FACTOR

# Copy the essential functions inline for the smoke test
ONELOOP = ONE_LOOP_FACTOR
TWOLOOP = TWO_LOOP_FACTOR
MZ = 91.1876; M_3TEV = 3.0e3; DT = np.log(M_3TEV / MZ)

YB_C=1.630e-2; YS_C=3.06e-4; YD_C=1.54e-5
YT_C=0.967; YC_C=3.56e-3; YU_C=7.04e-6
G1_C=0.461228; G2_C=0.65096; G3_C=1.2123
YTAU_MZ=0.99378e-2; YMU_MZ=5.85042e-4; YE_MZ=2.77713e-6
VTS2=0.0415**2; VTD2=0.0087**2

def Q_inv(yb, ys, yd):
    zd=1.0/yd; zs=1.0/ys; zb=1.0/yb
    S=zd+zs+zb; R=np.sqrt(zd)+np.sqrt(zs)+np.sqrt(zb)
    return S/(R*R)

def compute_sensitivities(yb, ys, yd):
    zd=1.0/yd; zs=1.0/ys; zb=1.0/yb
    S=zd+zs+zb; R=np.sqrt(zd)+np.sqrt(zs)+np.sqrt(zb); R3=R*R*R; Q=S/(R*R)
    dQ_dzd=(R-S/np.sqrt(zd))/R3; dQ_dzs=(R-S/np.sqrt(zs))/R3; dQ_dzb=(R-S/np.sqrt(zb))/R3
    return -zd*dQ_dzd/Q, -zs*dQ_dzs/Q, -zb*dQ_dzb/Q

def initial_conditions(yb=YB_C, ys=YS_C, yd=YD_C, yt=YT_C, yc=YC_C, yu=YU_C, g1=G1_C, g2=G2_C, g3=G3_C):
    return np.array([g1,g2,g3,yt,yc,yu,yb,ys,yd,YTAU_MZ,YMU_MZ,YE_MZ])

def beta_1loop_ckm(t, y):
    g1,g2,g3,yt,yc,yu,yb,ys,yd,ytau,ymu,ye=y
    Tr_Yu2=yt*yt+yc*yc+yu*yu; Tr_Yd2=yb*yb+ys*ys+yd*yd
    Tr_Ye2=ytau*ytau+ymu*ymu+ye*ye; T=3.0*Tr_Yu2+3.0*Tr_Yd2+Tr_Ye2
    g1s,g2s,g3s=g1*g1,g2*g2,g3*g3
    b1,b2,b3=41.0/10.0,-19.0/6.0,-7.0
    dg1=ONELOOP*b1*g1*g1s; dg2=ONELOOP*b2*g2*g2s; dg3=ONELOOP*b3*g3*g3s
    gauge_up=-(17.0/20.0)*g1s-(9.0/4.0)*g2s-8.0*g3s
    gauge_down=-(1.0/4.0)*g1s-(9.0/4.0)*g2s-8.0*g3s
    gauge_lep=-(9.0/4.0)*g1s-(9.0/4.0)*g2s
    dyt=ONELOOP*yt*(1.5*yt*yt-1.5*yb*yb+T+gauge_up)
    dyc=ONELOOP*yc*(1.5*yc*yc-1.5*ys*ys+T+gauge_up)
    dyu=ONELOOP*yu*(1.5*yu*yu-1.5*yd*yd+T+gauge_up)
    up_b=yt*yt; up_s=yc*yc+VTS2*yt*yt; up_d=yu*yu+VTD2*yt*yt
    dyb=ONELOOP*yb*(1.5*yb*yb-1.5*up_b+T+gauge_down)
    dys=ONELOOP*ys*(1.5*ys*ys-1.5*up_s+T+gauge_down)
    dyd=ONELOOP*yd*(1.5*yd*yd-1.5*up_d+T+gauge_down)
    dytau=ONELOOP*ytau*(1.5*ytau*ytau+T+gauge_lep)
    dymu=ONELOOP*ymu*(1.5*ymu*ymu+T+gauge_lep)
    dye=ONELOOP*ye*(1.5*ye*ye+T+gauge_lep)
    return [dg1,dg2,dg3,dyt,dyc,dyu,dyb,dys,dyd,dytau,dymu,dye]

def beta_2loop_up_only(t, y):
    d1 = beta_1loop_ckm(t, y)
    g1,g2,g3,yt,yc,yu,yb,ys,yd,ytau,ymu,ye=y
    g1s,g2s,g3s=g1*g1,g2*g2,g3*g3; g1q,g2q,g3q=g1s*g1s,g2s*g2s,g3s*g3s
    yts,ycs,yus=yt*yt,yc*yc,yu*yu; ybs,yss,yds=yb*yb,ys*ys,yd*yd
    Tr_Yu2=yts+ycs+yus; Tr_Yd2=ybs+yss+yds; Tr_Ye2=ytau*ytau+ymu*ymu+ye*ye
    T=3.0*Tr_Yu2+3.0*Tr_Yd2+Tr_Ye2
    B11,B12,B13=199.0/50.0,27.0/10.0,44.0/5.0
    B21,B22,B23=9.0/10.0,35.0/6.0,12.0
    B31,B32,B33=11.0/10.0,9.0/2.0,-26.0
    S1=B11*g1s+B12*g2s+B13*g3s; S2=B21*g1s+B22*g2s+B23*g3s; S3=B31*g1s+B32*g2s+B33*g3s
    Y1=(17.0/10.0)*Tr_Yu2+(1.0/2.0)*Tr_Yd2+(3.0/2.0)*Tr_Ye2
    Y2=(3.0/2.0)*Tr_Yu2+(3.0/2.0)*Tr_Yd2+(1.0/2.0)*Tr_Ye2
    Y3=2.0*Tr_Yu2+2.0*Tr_Yd2
    dg1_2L=TWOLOOP*g1*g1s*(S1-Y1); dg2_2L=TWOLOOP*g2*g2s*(S2-Y2); dg3_2L=TWOLOOP*g3*g3s*(S3-Y3)
    pure_gauge_up=((1187.0/600.0)*g1q-(9.0/20.0)*g1s*g2s+(19.0/15.0)*g1s*g3s-(23.0/4.0)*g2q+9.0*g2s*g3s-108.0*g3q)
    gauge_HH_yt=(223.0/80.0)*g1s*yts+(135.0/16.0)*g2s*yts+16.0*g3s*yts
    gauge_HH_yc=(223.0/80.0)*g1s*ycs+(135.0/16.0)*g2s*ycs+16.0*g3s*ycs
    gauge_HH_yu=(223.0/80.0)*g1s*yus+(135.0/16.0)*g2s*yus+16.0*g3s*yus
    gauge_FDFD_yt=-(43.0/80.0)*g1s*ybs+(9.0/16.0)*g2s*ybs-16.0*g3s*ybs
    gauge_FDFD_yc=-(43.0/80.0)*g1s*yss+(9.0/16.0)*g2s*yss-16.0*g3s*yss
    gauge_FDFD_yu=-(43.0/80.0)*g1s*yds+(9.0/16.0)*g2s*yds-16.0*g3s*yds
    gauge_traces=((17.0/8.0)*g1s*Tr_Yu2+(45.0/8.0)*g2s*Tr_Yu2+20.0*g3s*Tr_Yu2+(5.0/8.0)*g1s*Tr_Yd2+(45.0/8.0)*g2s*Tr_Yd2+20.0*g3s*Tr_Yd2+(15.0/8.0)*g1s*Tr_Ye2+(15.0/8.0)*g2s*Tr_Ye2)
    Tr_H4=yts*yts+ycs*ycs+yus*yus; Tr_FD4=ybs*ybs+yss*yss+yds*yds; Tr_FL4=ytau**4+ymu**4+ye**4
    Tr_HH_FDFD=2.0*(yts*ybs+ycs*yss+yus*yds)
    chi4=(9.0/4.0)*(3.0*Tr_H4+3.0*Tr_FD4+Tr_FL4-(1.0/3.0)*Tr_HH_FDFD)
    lam=0.126; lam2=lam*lam
    lam_HH_yt=(3.0/2.0)*lam2-6.0*lam*yts; lam_HH_yc=(3.0/2.0)*lam2-6.0*lam*ycs; lam_HH_yu=(3.0/2.0)*lam2-6.0*lam*yus
    common_2L=pure_gauge_up+gauge_traces-chi4
    yt_pure=((3.0/2.0)*yts*yts-yts*ybs-(1.0/4.0)*yts*ybs+(11.0/4.0)*ybs*ybs-(9.0/4.0)*T*yts+(5.0/4.0)*T*ybs)
    yc_pure=((3.0/2.0)*ycs*ycs-ycs*yss-(1.0/4.0)*ycs*yss+(11.0/4.0)*yss*yss-(9.0/4.0)*T*ycs+(5.0/4.0)*T*yss)
    yu_pure=((3.0/2.0)*yus*yus-yus*yds-(1.0/4.0)*yus*yds+(11.0/4.0)*yds*yds-(9.0/4.0)*T*yus+(5.0/4.0)*T*yds)
    dyt_2L=TWOLOOP*yt*(common_2L+gauge_HH_yt+gauge_FDFD_yt+yt_pure+lam_HH_yt)
    dyc_2L=TWOLOOP*yc*(common_2L+gauge_HH_yc+gauge_FDFD_yc+yc_pure+lam_HH_yc)
    dyu_2L=TWOLOOP*yu*(common_2L+gauge_HH_yu+gauge_FDFD_yu+yu_pure+lam_HH_yu)
    return [d1[0]+dg1_2L,d1[1]+dg2_2L,d1[2]+dg3_2L,d1[3]+dyt_2L,d1[4]+dyc_2L,d1[5]+dyu_2L,d1[6],d1[7],d1[8],d1[9],d1[10],d1[11]]

def beta_2loop_full_ckm(t, y):
    d1 = beta_1loop_ckm(t, y)
    g1,g2,g3,yt,yc,yu,yb,ys,yd,ytau,ymu,ye=y
    g1s,g2s,g3s=g1*g1,g2*g2,g3*g3; g1q,g2q,g3q=g1s*g1s,g2s*g2s,g3s*g3s
    yts,ycs,yus=yt*yt,yc*yc,yu*yu; ybs,yss,yds=yb*yb,ys*ys,yd*yd
    Tr_Yu2=yts+ycs+yus; Tr_Yd2=ybs+yss+yds; Tr_Ye2=ytau*ytau+ymu*ymu+ye*ye
    T=3.0*Tr_Yu2+3.0*Tr_Yd2+Tr_Ye2
    B11,B12,B13=199.0/50.0,27.0/10.0,44.0/5.0
    B21,B22,B23=9.0/10.0,35.0/6.0,12.0
    B31,B32,B33=11.0/10.0,9.0/2.0,-26.0
    S1=B11*g1s+B12*g2s+B13*g3s; S2=B21*g1s+B22*g2s+B23*g3s; S3=B31*g1s+B32*g2s+B33*g3s
    Y1=(17.0/10.0)*Tr_Yu2+(1.0/2.0)*Tr_Yd2+(3.0/2.0)*Tr_Ye2
    Y2=(3.0/2.0)*Tr_Yu2+(3.0/2.0)*Tr_Yd2+(1.0/2.0)*Tr_Ye2; Y3=2.0*Tr_Yu2+2.0*Tr_Yd2
    dg1_2L=TWOLOOP*g1*g1s*(S1-Y1); dg2_2L=TWOLOOP*g2*g2s*(S2-Y2); dg3_2L=TWOLOOP*g3*g3s*(S3-Y3)
    pure_gauge_up=((1187.0/600.0)*g1q-(9.0/20.0)*g1s*g2s+(19.0/15.0)*g1s*g3s-(23.0/4.0)*g2q+9.0*g2s*g3s-108.0*g3q)
    gauge_HH_yt=(223.0/80.0)*g1s*yts+(135.0/16.0)*g2s*yts+16.0*g3s*yts
    gauge_HH_yc=(223.0/80.0)*g1s*ycs+(135.0/16.0)*g2s*ycs+16.0*g3s*ycs
    gauge_HH_yu=(223.0/80.0)*g1s*yus+(135.0/16.0)*g2s*yus+16.0*g3s*yus
    gauge_FDFD_yt=-(43.0/80.0)*g1s*ybs+(9.0/16.0)*g2s*ybs-16.0*g3s*ybs
    gauge_FDFD_yc=-(43.0/80.0)*g1s*yss+(9.0/16.0)*g2s*yss-16.0*g3s*yss
    gauge_FDFD_yu=-(43.0/80.0)*g1s*yds+(9.0/16.0)*g2s*yds-16.0*g3s*yds
    gauge_traces=((17.0/8.0)*g1s*Tr_Yu2+(45.0/8.0)*g2s*Tr_Yu2+20.0*g3s*Tr_Yu2+(5.0/8.0)*g1s*Tr_Yd2+(45.0/8.0)*g2s*Tr_Yd2+20.0*g3s*Tr_Yd2+(15.0/8.0)*g1s*Tr_Ye2+(15.0/8.0)*g2s*Tr_Ye2)
    Tr_H4=yts*yts+ycs*ycs+yus*yus; Tr_FD4=ybs*ybs+yss*yss+yds*yds; Tr_FL4=ytau**4+ymu**4+ye**4
    Tr_HH_FDFD=2.0*(yts*ybs+ycs*yss+yus*yds)
    chi4=(9.0/4.0)*(3.0*Tr_H4+3.0*Tr_FD4+Tr_FL4-(1.0/3.0)*Tr_HH_FDFD)
    lam=0.126; lam2=lam*lam
    lam_HH_yt=(3.0/2.0)*lam2-6.0*lam*yts; lam_HH_yc=(3.0/2.0)*lam2-6.0*lam*ycs; lam_HH_yu=(3.0/2.0)*lam2-6.0*lam*yus
    common_2L=pure_gauge_up+gauge_traces-chi4
    yt_pure=((3.0/2.0)*yts*yts-(5.0/4.0)*yts*ybs+(11.0/4.0)*ybs*ybs-(9.0/4.0)*T*yts+(5.0/4.0)*T*ybs)
    yc_pure=((3.0/2.0)*ycs*ycs-(5.0/4.0)*ycs*yss+(11.0/4.0)*yss*yss-(9.0/4.0)*T*ycs+(5.0/4.0)*T*yss)
    yu_pure=((3.0/2.0)*yus*yus-(5.0/4.0)*yus*yds+(11.0/4.0)*yds*yds-(9.0/4.0)*T*yus+(5.0/4.0)*T*yds)
    dyt_2L=TWOLOOP*yt*(common_2L+gauge_HH_yt+gauge_FDFD_yt+yt_pure+lam_HH_yt)
    dyc_2L=TWOLOOP*yc*(common_2L+gauge_HH_yc+gauge_FDFD_yc+yc_pure+lam_HH_yc)
    dyu_2L=TWOLOOP*yu*(common_2L+gauge_HH_yu+gauge_FDFD_yu+yu_pure+lam_HH_yu)
    # Down-type 2-loop WITH CKM
    gauge_FDFD_yb=(223.0/80.0)*g1s*ybs+(135.0/16.0)*g2s*ybs+16.0*g3s*ybs
    gauge_FDFD_ys=(223.0/80.0)*g1s*yss+(135.0/16.0)*g2s*yss+16.0*g3s*yss
    gauge_FDFD_yd=(223.0/80.0)*g1s*yds+(135.0/16.0)*g2s*yds+16.0*g3s*yds
    cross_coeff=-(43.0/80.0)*g1s+(9.0/16.0)*g2s-16.0*g3s
    gauge_HH_yb_ckm=cross_coeff*yts
    gauge_HH_ys_ckm=cross_coeff*(ycs+VTS2*yts)
    gauge_HH_yd_ckm=cross_coeff*(yus+VTD2*yts)
    up_eff_b=yts; up_eff_s=ycs+VTS2*yts; up_eff_d=yus+VTD2*yts
    yb_pure_ckm=((3.0/2.0)*ybs*ybs-(5.0/4.0)*ybs*up_eff_b+(11.0/4.0)*up_eff_b*up_eff_b-(9.0/4.0)*T*ybs+(5.0/4.0)*T*up_eff_b)
    ys_pure_ckm=((3.0/2.0)*yss*yss-(5.0/4.0)*yss*up_eff_s+(11.0/4.0)*up_eff_s*up_eff_s-(9.0/4.0)*T*yss+(5.0/4.0)*T*up_eff_s)
    yd_pure_ckm=((3.0/2.0)*yds*yds-(5.0/4.0)*yds*up_eff_d+(11.0/4.0)*up_eff_d*up_eff_d-(9.0/4.0)*T*yds+(5.0/4.0)*T*up_eff_d)
    lam_FDFD_yb=(3.0/2.0)*lam2-6.0*lam*ybs; lam_FDFD_ys=(3.0/2.0)*lam2-6.0*lam*yss; lam_FDFD_yd=(3.0/2.0)*lam2-6.0*lam*yds
    dyb_2L=TWOLOOP*yb*(common_2L+gauge_FDFD_yb+gauge_HH_yb_ckm+yb_pure_ckm+lam_FDFD_yb)
    dys_2L=TWOLOOP*ys*(common_2L+gauge_FDFD_ys+gauge_HH_ys_ckm+ys_pure_ckm+lam_FDFD_ys)
    dyd_2L=TWOLOOP*yd*(common_2L+gauge_FDFD_yd+gauge_HH_yd_ckm+yd_pure_ckm+lam_FDFD_yd)
    return [d1[0]+dg1_2L,d1[1]+dg2_2L,d1[2]+dg3_2L,d1[3]+dyt_2L,d1[4]+dyc_2L,d1[5]+dyu_2L,d1[6]+dyb_2L,d1[7]+dys_2L,d1[8]+dyd_2L,d1[9],d1[10],d1[11]]

def evolve_one(y0, beta_func, t_max=DT):
    sol=solve_ivp(beta_func,[0.0,t_max],y0,method='RK45',rtol=1e-10,atol=1e-12,t_eval=np.linspace(0.0,t_max,20),max_step=0.5)
    return sol if sol.success else None

# ─── RUN SMOKE TEST ───
print("="*60)
print("SMOKE TEST: down2L central-value computations")
print(f"M_Z={MZ}, M_3TeV={M_3TEV}, DT={DT:.4f}")
print(f"yb={YB_C:.6e}, ys={YS_C:.6e}, yd={YD_C:.6e}, yt={YT_C}")
print(f"|V_ts|²={VTS2:.6f}, |V_td|²={VTD2:.6f}")
print("="*60)

y0 = initial_conditions()
q_mz_0 = Q_inv(YB_C, YS_C, YD_C)
print(f"\nQ_inv(M_Z) = {q_mz_0:.8f}")

# Sensitivities
sd, ss, sb = compute_sensitivities(YB_C, YS_C, YD_C)
print(f"Sensitivities dlnQ/dln[y_d,y_s,y_b] = [{sd:+.4f}, {ss:+.4f}, {sb:+.4f}]")
print(f"Reference values:                 [-0.155, +0.131, +0.024]")

# 1-loop
print("\n--- (a) 1-loop ---")
sol = evolve_one(y0, beta_1loop_ckm)
if sol:
    qmz, q3, dabs, dpct = Q_inv(YB_C,YS_C,YD_C), Q_inv(float(np.interp(DT,sol.t,sol.y[6])),float(np.interp(DT,sol.t,sol.y[7])),float(np.interp(DT,sol.t,sol.y[8]))), 0, 0
    dabs = q3-qmz; dpct = 100.0*dabs/qmz
    print(f"  Q_inv(3TeV) = {q3:.8f}, drift = {dpct:+.4f}%")
else:
    print("  FAILED")

# 2-loop up-only
print("\n--- (b) 2-loop up-only ---")
sol = evolve_one(y0, beta_2loop_up_only)
if sol:
    q3 = Q_inv(float(np.interp(DT,sol.t,sol.y[6])),float(np.interp(DT,sol.t,sol.y[7])),float(np.interp(DT,sol.t,sol.y[8])))
    dpct = 100.0*(q3-qmz)/qmz
    print(f"  Q_inv(3TeV) = {q3:.8f}, drift = {dpct:+.4f}%")
else:
    print("  FAILED")

# 2-loop full CKM
print("\n--- (c) 2-loop full (CKM) ---")
sol = evolve_one(y0, beta_2loop_full_ckm)
if sol:
    yb_f=float(np.interp(DT,sol.t,sol.y[6])); ys_f=float(np.interp(DT,sol.t,sol.y[7])); yd_f=float(np.interp(DT,sol.t,sol.y[8]))
    q3 = Q_inv(yb_f, ys_f, yd_f)
    dpct = 100.0*(q3-qmz)/qmz
    print(f"  Q_inv(3TeV) = {q3:.8f}, drift = {dpct:+.4f}%")
    print(f"  yb: {YB_C:.6e} -> {yb_f:.6e}  ({100*(yb_f/YB_C-1):+.3f}%)")
    print(f"  ys: {YS_C:.6e} -> {ys_f:.6e}  ({100*(ys_f/YS_C-1):+.3f}%)")
    print(f"  yd: {YD_C:.6e} -> {yd_f:.6e}  ({100*(yd_f/YD_C-1):+.3f}%)")
else:
    print("  FAILED")

# Compare with gate_D1R values
print("\n--- COMPARISON with gate_D1R ---")
print("  gate_D1R: 1L drift = -0.0624%, 2L drift = -0.0650%, trunc = 0.0025%")
print("  (gate_D1R uses uncorrected sm_rge beta_1loop/beta_2loop, no CKM)")
print("  Measured: -0.0779% ± 0.0006%")

# Time a single integration
import time
tic=time.time()
for _ in range(10):
    evolve_one(y0, beta_1loop_ckm)
t1L = (time.time()-tic)/10
tic=time.time()
for _ in range(10):
    evolve_one(y0, beta_2loop_full_ckm)
t2L = (time.time()-tic)/10
print(f"\n  Timing: 1L={t1L:.3f}s, 2L_full={t2L:.3f}s per integration")
print(f"  Estimated for N=100k × 3 integrators: {(t1L+t2L+t2L)*100000/3600:.1f} hours")
print("\nDONE.")
