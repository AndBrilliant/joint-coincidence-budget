#!/usr/bin/env python3
"""
v0.5 TIER-2 VALIDATION — Optimized with cached survivors and batched anchor averaging.
ENGINE_ID: amb | Calibration seed: 271828

Optimizations over the main engine:
1. Cache lepton-stage survivors across new-claim factors
2. Batch anchor averaging: process multiple lepton samples per quark draw batch
3. Early termination: stop at first factor that reaches 100 hits
4. Pre-compute non-vacuous gate decisions
"""

import numpy as np
import json
import os
import sys
import time
from scipy.stats import beta as beta_dist

# ── Import shared functions from main engine ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
# Copy-paste essential functions to avoid import issues with dotted filename

# ═══════════════════════════════════════════════════════════════════════════
# KDISK, Q_U, CP95
# ═══════════════════════════════════════════════════════════════════════════

ang = 2.0 * np.pi * np.arange(3) / 3.0
cos_ang = np.cos(ang); sin_ang = np.sin(ang)

def kdist(m):
    m = np.asarray(m)
    out = None
    for v in (np.sqrt(m), 1.0 / np.sqrt(m)):
        A = v.mean(axis=-1)
        X = (2.0/3.0)*(v*cos_ang).sum(axis=-1)
        Y = -(2.0/3.0)*(v*sin_ang).sum(axis=-1)
        d = np.abs(np.hypot(X,Y)/(np.sqrt(2.0)*A)-1.0)
        out = d if out is None else np.minimum(out, d)
    return out

def Q_U(v):
    v = np.asarray(v)
    return np.sum(v, axis=-1) / np.sum(np.sqrt(v), axis=-1)**2

def cp95(k, n, alpha=0.05):
    if k <= 0: lower = 0.0
    else: lower = beta_dist.ppf(alpha/2.0, k, n-k+1)
    if k >= n: upper = 1.0
    else: upper = beta_dist.ppf(1.0-alpha/2.0, k+1, n-k)
    return float(lower), float(upper)

# ═══════════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

LEPTONS_OBS = np.array([0.51099895, 105.6583755, 1776.93])
L1_TOL, L2_TOL, L3_TOL = 3.3049e-6, 1.00e-5, 2.10e-5
L2_TARGET, L3_TARGET = 206.7703, 16.8180
B1, B2 = 3.00e-3, 1.18e-2
U1_TOL = 1.1414e-2

U1_MENU_TARGETS = np.array([9.0*p/q for (p,q) in [
    (1,1),(1,2),(2,3),(3,4),(2,5),(3,5),(4,5),(5,6),
    (3,7),(4,7),(5,7),(6,7),(3,8),(5,8),(7,8),(4,9),(5,9),(7,9),(8,9)
]], dtype=np.float64)

LEP_LO, LEP_HI = 0.3, 2000.0
QUARK_LO, QUARK_HI = 0.5, 2e5
LEP_LOG_LO = np.log(LEP_LO); LEP_LOG_HI = np.log(LEP_HI)
LEP_LOG_V = LEP_LOG_HI - LEP_LOG_LO
QUARK_LOG_LO = np.log(QUARK_LO); QUARK_LOG_HI = np.log(QUARK_HI)
QUARK_LOG_V = QUARK_LOG_HI - QUARK_LOG_LO
HIERARCHY_MIN = (4.0 + np.sqrt(18.0))**2
INFLATION_FACTORS = [1, 3, 10, 30, 100, 300, 1000]

# ═══════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS (with factor)
# ═══════════════════════════════════════════════════════════════════════════

def chk_L1(lep, f=1.0): return kdist(lep) <= L1_TOL*f
def chk_L2(lep, f=1.0): return np.abs(lep[:,1]/lep[:,0]/L2_TARGET-1.0) <= L2_TOL*f
def chk_L3(lep, f=1.0): return np.abs(lep[:,2]/lep[:,1]/L3_TARGET-1.0) <= L3_TOL*f
def chk_Q1(lq, lep, f=1.0):
    mu,md,ms = lq[:,0],lq[:,1],lq[:,2]; mu_s = lep.sum(axis=1)
    return np.abs(np.log(ms*ms/(mu_s*md))) <= B1*f
def chk_Q2(lq, lep, f=1.0):
    mu,md = lq[:,0],lq[:,1]; tw = 2.0*lep.min(axis=1)
    return np.abs(np.log(mu*mu/(md*tw))) <= B2*f
def chk_U1f(lq, us, f=1.0):
    t = np.column_stack([lq[:,0],us[:,0],us[:,1]])
    qd,qi = Q_U(t),Q_U(1.0/t)
    return np.minimum(np.abs(9.0*qd-8.0),np.abs(9.0*qi-8.0)) <= U1_TOL*f
def chk_U1m(lq, us, f=1.0):
    t = np.column_stack([lq[:,0],us[:,0],us[:,1]])
    qd,qi = Q_U(t),Q_U(1.0/t)
    hit = np.zeros(len(lq[:,0]), dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9.0*qd-tgt) <= U1_TOL*f)
        hit |= (np.abs(9.0*qi-tgt) <= U1_TOL*f)
    return hit

# ═══════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_logu_triple(rng, N, lo, hi, sort=True):
    x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(N,3)))
    if sort: x.sort(axis=1)
    return x

def draw_logu_pair(rng, N, lo, hi):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=(N,2)))

def sample_t1_koide(rng, batch_size):
    m3 = np.exp(rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r*m3; s1,s3 = np.sqrt(m1),np.sqrt(m3)
    b = -4.0*(s1+s3); c = s1**2+s3**2-4.0*s1*s3
    disc = b**2-4.0*c; vd = disc>=0; disc=np.maximum(disc,0.0)
    s2 = (-b-np.sqrt(disc))/2.0
    s2ok = s2>0; sok = (m1<s2**2)&(s2**2<m3); hok = (m3/m1)>HIERARCHY_MIN
    keep = vd&s2ok&sok&hok; m2=s2[keep]**2
    return np.column_stack([m1[keep],m2,m3[keep]]), batch_size

# ═══════════════════════════════════════════════════════════════════════════
# NON-VACUOUS GATE (quick)
# ═══════════════════════════════════════════════════════════════════════════

def nonvacuous(claim, factor):
    if claim in ("L2","L3"):
        V=LEP_LOG_V; t=np.log(L2_TARGET) if claim=="L2" else np.log(L3_TARGET)
        tol=L2_TOL if claim=="L2" else L3_TOL
        hw=tol*factor; lo,hi=max(0.0,t-hw),min(V,t+hw)
        if lo>=hi: return True,0.0
        F=lambda x:(2*x/V)-(x/V)**2
        frac=F(hi)-F(lo); return frac<0.5, float(max(0,frac))
    elif claim=="L1":
        rng=np.random.default_rng(271828); N=200000
        x=rng.uniform(LEP_LOG_LO,LEP_LOG_HI,size=(N,3)); x.sort(axis=1)
        m=np.exp(x); frac=chk_L1(m,factor).mean(); return frac<0.5,float(frac)
    elif claim in ("Q1","Q2"):
        tol=B1 if claim=="Q1" else B2
        frac=2.0*tol*factor/QUARK_LOG_V; return frac<0.5,float(min(frac,1.0))
    elif claim.startswith("U1"):
        rng=np.random.default_rng(271828); N=100000
        lq=draw_logu_triple(rng,N,QUARK_LO,QUARK_HI)
        us=draw_logu_pair(rng,N,QUARK_LO,QUARK_HI)
        if "fixed" in claim: frac=chk_U1f(lq,us,factor).mean()
        else: frac=chk_U1m(lq,us,factor).mean()
        return frac<0.5,float(frac)
    return True,0.0

# ═══════════════════════════════════════════════════════════════════════════
# TIER-2 LEPTON BLOCK (analytic)
# ═══════════════════════════════════════════════════════════════════════════

def lepton_block_prob(condition, lep_factors):
    """Analytic P(lepton stage) at given inflation factors."""
    if condition=="T0":
        f1,f2,f3 = lep_factors.get("L1",1.0), lep_factors.get("L2",1.0), lep_factors.get("L3",1.0)
        u_lo=np.log(L2_TARGET*(1.0-L2_TOL*f2)); u_hi=np.log(L2_TARGET*(1.0+L2_TOL*f2))
        v_lo=np.log(L3_TARGET*(1.0-L3_TOL*f3)); v_hi=np.log(L3_TARGET*(1.0+L3_TOL*f3))
        u_mid=(u_lo+u_hi)/2.0; v_mid=(v_lo+v_hi)/2.0; du=u_hi-u_lo; dv=v_hi-v_lo
        r_val = LEP_LOG_V-u_mid-v_mid
        f0 = 6.0/(LEP_LOG_V**3)*r_val if r_val>0 else 0.0
        # L1 fraction
        N_l1=50000; rng=np.random.default_rng(271828)
        u_t=rng.uniform(u_lo,u_hi,N_l1); v_t=rng.uniform(v_lo,v_hi,N_l1)
        m1=np.ones(N_l1); m2=m1*np.exp(u_t); m3=m2*np.exp(v_t)
        lt=np.column_stack([m1,m2,m3]); fl1=float(chk_L1(lt,f1).mean())
        return f0*du*dv*fl1
    else:  # T1
        f2,f3 = lep_factors.get("L2",1.0), lep_factors.get("L3",1.0)
        # Quick r-intersection via grid
        r_obs=0.51099895/1776.93; lnr=np.log(r_obs)
        for spread in [1e-4,1e-3,1e-2,1e-1,1.0]:
            rg=np.exp(lnr+np.linspace(-spread,spread,200000))
            x=np.sqrt(np.maximum(rg,1e-300))
            disc=3.0*(x**2+4.0*x+1.0)
            s2s3=2.0*(x+1.0)-np.sqrt(np.maximum(0.0,disc))
            m2m1=s2s3**2/x**2; m3m2=1.0/(s2s3**2)
            l2ok=np.abs(m2m1/L2_TARGET-1.0)<=L2_TOL*max(f2,f3)
            l3ok=np.abs(m3m2/L3_TARGET-1.0)<=L3_TOL*max(f2,f3)
            both=l2ok&l3ok
            if both.any():
                ri=rg[both]; dlr=np.log(ri.max()/ri.min())
                if dlr>0: return dlr/np.log(1e-1/1e-5)
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
# BATCHED ANCHOR-AVERAGED QUARK BLOCK (optimized)
# ═══════════════════════════════════════════════════════════════════════════

def quark_block_anchored_batched(lepton_samples, quark_claims, factor, u1_mode,
                                  N_quark_per_batch=100000, n_batches=50, seed=271828):
    """Batched anchor averaging: process quarks once, check against all lepton samples.

    Strategy: draw a large quark pool, check all lepton samples against it,
    then average. Repeat for multiple batches to reduce variance.

    Returns: mean p_quark, std error
    """
    N_lep = len(lepton_samples)
    anchors_mu_star = lepton_samples.sum(axis=1)
    anchors_twome = 2.0 * lepton_samples.min(axis=1)

    rng = np.random.default_rng(seed)
    batch_means = []

    for b in range(n_batches):
        # Draw quark pool
        lq = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_quark_per_batch, 3)))
        lq.sort(axis=1)
        us = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_quark_per_batch, 2)))

        # For each lepton sample, compute hit rate against this quark pool
        lep_hit_rates = np.zeros(N_lep)
        for i in range(N_lep):
            mu_s = anchors_mu_star[i]
            tw = anchors_twome[i]
            surv = np.ones(N_quark_per_batch, dtype=bool)

            for claim in quark_claims:
                if claim == "Q1":
                    mu,md,ms = lq[:,0],lq[:,1],lq[:,2]
                    surv &= (np.abs(np.log(ms*ms/(mu_s*md))) <= B1*factor)
                elif claim == "Q2":
                    mu,md = lq[:,0],lq[:,1]
                    surv &= (np.abs(np.log(mu*mu/(md*tw))) <= B2*factor)
                elif claim == "U1_fixed":
                    t = np.column_stack([mu, us[:,0], us[:,1]])
                    qd,qi = Q_U(t),Q_U(1.0/t)
                    surv &= (np.minimum(np.abs(9.0*qd-8.0),np.abs(9.0*qi-8.0)) <= U1_TOL*factor)
                elif claim == "U1_menu":
                    t = np.column_stack([mu, us[:,0], us[:,1]])
                    qd,qi = Q_U(t),Q_U(1.0/t)
                    hit = np.zeros(N_quark_per_batch, dtype=bool)
                    for tgt in U1_MENU_TARGETS:
                        hit |= (np.abs(9.0*qd-tgt) <= U1_TOL*factor)
                        hit |= (np.abs(9.0*qi-tgt) <= U1_TOL*factor)
                    surv &= hit

            lep_hit_rates[i] = surv.mean()

        batch_means.append(lep_hit_rates.mean())

    return float(np.mean(batch_means)), float(np.std(batch_means)/np.sqrt(n_batches))

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED PER-STAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_stage_optimized(condition, u1_mode, stage_name, lepton_claims, new_claims,
                              lepton_factor, calibration_seed=271828, N_max=1_000_000_000):
    """Validate one stage with cached lepton-stage survivors.

    Returns: (result_dict, validated_bool)
    """
    batch_size = 1_000_000
    stage_idx = {"+Q1":1, "+Q2":2, "+U1":3}.get(stage_name, 0)

    print(f"\n  Validating {stage_name} (lepton factor={lepton_factor})...")

    # ── Draw lepton samples for anchor averaging (once, cached) ──
    N_lep_samples = 10000
    print(f"  Drawing {N_lep_samples} lepton samples for anchor averaging...")
    rng_lep = np.random.default_rng(calibration_seed + 1000 + stage_idx)
    lep_samples_list = []

    if condition == "T0":
        f1,f2,f3 = lepton_factor,lepton_factor,lepton_factor
        u_lo=np.log(L2_TARGET*(1.0-L2_TOL*f2)); u_hi=np.log(L2_TARGET*(1.0+L2_TOL*f2))
        v_lo=np.log(L3_TARGET*(1.0-L3_TOL*f3)); v_hi=np.log(L3_TARGET*(1.0+L3_TOL*f3))
        while len(lep_samples_list) < N_lep_samples:
            u_s=rng_lep.uniform(u_lo,u_hi,batch_size)
            v_s=rng_lep.uniform(v_lo,v_hi,batch_size)
            m1=np.ones(batch_size); m2=m1*np.exp(u_s); m3=m2*np.exp(v_s)
            lc=np.column_stack([m1,m2,m3])
            ok=chk_L1(lc,f1)
            if ok.any(): lep_samples_list.append(lc[ok])
    else:  # T1
        # T1: use r-intersection for sampling
        r_obs=0.51099895/1776.93
        for spread in [1e-4,1e-3,1e-2,1e-1]:
            rg=np.exp(np.log(r_obs)+np.linspace(-spread,spread,200000))
            x=np.sqrt(np.maximum(rg,1e-300))
            disc=3.0*(x**2+4.0*x+1.0); s2s3=2.0*(x+1.0)-np.sqrt(np.maximum(0.0,disc))
            m2m1=s2s3**2/x**2; m3m2=1.0/(s2s3**2)
            l2ok=np.abs(m2m1/L2_TARGET-1.0)<=L2_TOL*lepton_factor
            l3ok=np.abs(m3m2/L3_TARGET-1.0)<=L3_TOL*lepton_factor
            both=l2ok&l3ok
            if both.any():
                ri=rg[both]; r_min,r_max=ri.min(),ri.max()
                break
        while len(lep_samples_list) < N_lep_samples:
            r_s=np.exp(rng_lep.uniform(np.log(r_min),np.log(r_max),batch_size))
            m3_s=np.exp(rng_lep.uniform(LEP_LOG_LO,LEP_LOG_HI,batch_size))
            m1_s=r_s*m3_s; s1,s3=np.sqrt(m1_s),np.sqrt(m3_s)
            b_s=-4.0*(s1+s3); c_s=s1**2+s3**2-4.0*s1*s3
            disc_s=b_s**2-4.0*c_s; vd=disc_s>=0; disc_s=np.maximum(disc_s,0.0)
            s2_s=(-b_s-np.sqrt(disc_s))/2.0
            s2ok=s2_s>0; sok=(m1_s<s2_s**2)&(s2_s**2<m3_s); hok=(m3_s/m1_s)>HIERARCHY_MIN
            keep=vd&s2ok&sok&hok
            if keep.any():
                m2_s=s2_s[keep]**2
                lep_samples_list.append(np.column_stack([m1_s[keep],m2_s,m3_s[keep]]))

    lepton_samples = np.vstack(lep_samples_list)[:N_lep_samples]
    print(f"  Got {len(lepton_samples)} lepton samples")

    # ── Pre-generate lepton-stage survivors for brute force (cached) ──
    print(f"  Generating lepton-stage survivors at factor {lepton_factor}...")
    rng_bf = np.random.default_rng(calibration_seed + stage_idx * 7777)
    cached_survivors = []
    cached_survivors_lep = []
    total_eff = 0

    while len(cached_survivors) < 10000 and total_eff < N_max:  # Pool 10k survivors
        if condition == "T0":
            lep = draw_logu_triple(rng_bf, batch_size, LEP_LO, LEP_HI)
            n_acc = batch_size; total_eff += batch_size
        else:
            lep, attempted = sample_t1_koide(rng_bf, batch_size)
            total_eff += attempted; n_acc = len(lep)
            if n_acc == 0: continue

        surv = np.ones(n_acc, dtype=bool)
        for claim in lepton_claims:
            if claim=="L1": m=chk_L1(lep,lepton_factor)
            elif claim=="L2": m=chk_L2(lep,lepton_factor)
            elif claim=="L3": m=chk_L3(lep,lepton_factor)
            si=np.where(surv)[0]; surv[si[~m[si]]]=False
        if surv.any():
            cached_survivors.append(lep[surv])

    cached_lep = np.vstack(cached_survivors)
    N_cached = len(cached_lep)
    print(f"  Pool: {N_cached} lepton-stage survivors from {total_eff:,} draws")

    # ── Try new-claim factors ──
    lep_factors = {c: lepton_factor for c in lepton_claims}
    p_lep = lepton_block_prob(condition, lep_factors)

    for f_new in INFLATION_FACTORS:
        # Non-vacuous gate
        vacuous = False
        for claim in new_claims:
            nv, frac = nonvacuous(claim, f_new)
            if not nv:
                print(f"    Factor {f_new}: VACUOUS ({claim}: {frac*100:.1f}%)")
                vacuous = True; break
        if vacuous: continue

        print(f"    Factor {f_new}: computing...", end="", flush=True)

        # ── Analytic: anchor-averaged quark block ──
        t0_a = time.time()
        p_quark_avg, p_quark_err = quark_block_anchored_batched(
            lepton_samples, new_claims, f_new, u1_mode,
            N_quark_per_batch=50000, n_batches=20, seed=calibration_seed+stage_idx*10000
        )
        t2_analytic = p_lep * p_quark_avg
        ta = time.time() - t0_a

        # ── Brute force: use cached survivors ──
        t0_b = time.time()
        bf_hits = 0
        # Process cached survivors in batches
        for start in range(0, N_cached, 5000):
            end = min(start+5000, N_cached)
            lep_s = cached_lep[start:end]
            n_s = len(lep_s)

            lq = draw_logu_triple(rng_bf, n_s, QUARK_LO, QUARK_HI)
            us_bf = draw_logu_pair(rng_bf, n_s, QUARK_LO, QUARK_HI)
            surv_q = np.ones(n_s, dtype=bool)

            for claim in new_claims:
                if claim=="Q1": m=chk_Q1(lq,lep_s,f_new)
                elif claim=="Q2": m=chk_Q2(lq,lep_s,f_new)
                elif claim=="U1_fixed": m=chk_U1f(lq,us_bf,f_new)
                elif claim=="U1_menu": m=chk_U1m(lq,us_bf,f_new)
                si=np.where(surv_q)[0]; surv_q[si[~m[si]]]=False
            bf_hits += surv_q.sum()

        bf_rate = bf_hits / N_cached if N_cached > 0 else 0.0
        tb = time.time() - t0_b

        bf_sigma = np.sqrt(max(bf_hits,1)) / N_cached if N_cached > 0 else float('inf')
        dev = abs(t2_analytic - bf_rate)
        within = dev <= 2.0*bf_sigma
        dev_sigma = dev/bf_sigma if bf_sigma > 0 else float('inf')

        print(f" T2={t2_analytic:.4e}, BF={bf_rate:.4e}±{bf_sigma:.4e}, "
              f"{dev_sigma:.2f}σ, {'✓' if within else '✗'} "
              f"[anchor:{ta:.1f}s, bf:{tb:.1f}s]")

        if bf_hits >= 100:
            return {
                "new_claim_factor": f_new, "lepton_factor": lepton_factor,
                "t2_analytic": float(t2_analytic), "p_lep": float(p_lep),
                "p_quark_avg": float(p_quark_avg), "p_quark_err": float(p_quark_err),
                "bf_hits": int(bf_hits), "bf_N_cached": int(N_cached),
                "bf_rate": float(bf_rate), "bf_sigma": float(bf_sigma),
                "deviation_sigma": float(dev_sigma), "within_2sigma": bool(within),
                "total_eff": int(total_eff),
            }, bool(within)

    # No factor reached 100 hits
    return {"eligible": False, "best_hits": int(bf_hits) if 'bf_hits' in dir() else 0,
            "lepton_factor": lepton_factor}, False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="T1_menu_logU", help="Cell key to validate")
    p.add_argument("--outdir", default="results/amb-20260811-v0.5")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Parse cell key
    parts = args.cell.split("_")
    condition = "T0" if parts[0]=="T0" else "T1"
    u1_mode = "menu" if "menu" in args.cell else "fixed"
    prior = "logU"

    print(f"{'='*60}")
    print(f"v0.5 TIER-2 VALIDATION (optimized) — {args.cell}")
    print(f"Condition: {condition}, U1: {u1_mode}")
    print(f"{'='*60}")

    calibration_seed = 271828
    N_max = 1_000_000_000

    # Stage definitions
    if condition == "T0":
        lepton_claims_all = ["L1","L2","L3"]
        stages = [
            ("L1_L2_L3", ["L1","L2","L3"], []),
            ("+Q1", ["L1","L2","L3"], ["Q1"]),
            ("+Q2", ["L1","L2","L3"], ["Q2"]),
            ("+U1", ["L1","L2","L3"], [f"U1_{u1_mode}"]),
        ]
    else:
        lepton_claims_all = ["L2","L3"]
        stages = [
            ("L2_L3", ["L2","L3"], []),
            ("+Q1", ["L2","L3"], ["Q1"]),
            ("+Q2", ["L2","L3"], ["Q2"]),
            ("+U1", ["L2","L3"], [f"U1_{u1_mode}"]),
        ]

    # ── Step 1: Find lepton factor ──
    print(f"\nStep 1: Finding lepton-stage factor...")
    rng_bf = np.random.default_rng(calibration_seed)
    lepton_factor = None
    batch_size = 1_000_000

    for f_lep in INFLATION_FACTORS:
        vacuous = False
        for claim in lepton_claims_all:
            nv, frac = nonvacuous(claim, f_lep)
            if not nv:
                print(f"  Factor {f_lep}: VACUOUS ({claim}: {frac*100:.1f}%)")
                vacuous = True; break
        if vacuous: continue

        lep_hits = 0; lep_N = 0
        while lep_hits < 100 and lep_N < N_max:
            if condition == "T0":
                lep = draw_logu_triple(rng_bf, batch_size, LEP_LO, LEP_HI)
                n_acc = batch_size; lep_N += batch_size
            else:
                lep, attempted = sample_t1_koide(rng_bf, batch_size)
                lep_N += attempted; n_acc = len(lep)
                if n_acc == 0: continue
            surv = np.ones(n_acc, dtype=bool)
            for claim in lepton_claims_all:
                if claim=="L1": m=chk_L1(lep,f_lep)
                elif claim=="L2": m=chk_L2(lep,f_lep)
                elif claim=="L3": m=chk_L3(lep,f_lep)
                si=np.where(surv)[0]; surv[si[~m[si]]]=False
            lep_hits += surv.sum()

        print(f"  Factor {f_lep}: {lep_hits} survivors in {lep_N:,} draws")
        if lep_hits >= 100:
            lepton_factor = f_lep; break

    if lepton_factor is None:
        print("FATAL: No lepton factor reached 100 hits")
        return
    print(f"→ Lepton factor: {lepton_factor}")

    # ── Step 2: Validate each stage ──
    all_validated = True
    validation_record = {}

    for stage_name, lep_claims, new_claims in stages:
        if not new_claims:
            # Lepton-only stage: validate directly
            print(f"\n  Validating {stage_name} (lepton-only, factor={lepton_factor})...")
            rng_bf = np.random.default_rng(calibration_seed + 1)
            bf_hits = 0; bf_N = 0
            while bf_hits < 100 and bf_N < N_max:
                if condition == "T0":
                    lep = draw_logu_triple(rng_bf, batch_size, LEP_LO, LEP_HI)
                    n_acc = batch_size; bf_N += batch_size
                else:
                    lep, attempted = sample_t1_koide(rng_bf, batch_size)
                    bf_N += attempted; n_acc = len(lep)
                    if n_acc == 0: continue
                surv = np.ones(n_acc, dtype=bool)
                for claim in lep_claims:
                    if claim=="L1": m=chk_L1(lep,lepton_factor)
                    elif claim=="L2": m=chk_L2(lep,lepton_factor)
                    elif claim=="L3": m=chk_L3(lep,lepton_factor)
                    si=np.where(surv)[0]; surv[si[~m[si]]]=False
                bf_hits += surv.sum()

            bf_rate = bf_hits/bf_N if bf_N>0 else 0.0
            bf_sigma = np.sqrt(bf_hits)/bf_N if bf_N>0 else 0.0
            lep_factors = {c:lepton_factor for c in lep_claims}
            p_lep = lepton_block_prob(condition, lep_factors)
            dev = abs(p_lep-bf_rate)
            within = dev<=2.0*bf_sigma
            print(f"    T2={p_lep:.4e}, BF={bf_rate:.4e}±{bf_sigma:.4e}, "
                  f"{dev/bf_sigma:.2f}σ, {'✓' if within else '✗'}")
            validation_record[stage_name] = {"eligible":True, "lepton_factor":lepton_factor,
                "t2_analytic":float(p_lep), "bf_hits":int(bf_hits), "bf_N":int(bf_N),
                "bf_rate":float(bf_rate), "within_2sigma":bool(within)}
            if not within: all_validated = False
        else:
            # Has new claims
            if stage_name == "+U1":
                # Try direct first
                result, ok = validate_stage_optimized(
                    condition, u1_mode, stage_name, lep_claims, new_claims,
                    lepton_factor, calibration_seed, N_max
                )
                if result.get("eligible"):
                    validation_record[stage_name] = result
                    if not ok: all_validated = False
                else:
                    # Compositional validation
                    print(f"\n  +U1: Direct failed ({result.get('best_hits',0)} hits max). Trying compositional...")
                    # (i) P(U1) singleton
                    rng_u1 = np.random.default_rng(calibration_seed+55555)
                    N_sing = 10_000_000; u1_hits = 0; u1_N = 0
                    for _ in range(N_sing//100000):
                        lq=draw_logu_triple(rng_u1,100000,QUARK_LO,QUARK_HI)
                        us=draw_logu_pair(rng_u1,100000,QUARK_LO,QUARK_HI)
                        if u1_mode=="fixed": u1_hits+=chk_U1f(lq,us,1.0).sum()
                        else: u1_hits+=chk_U1m(lq,us,1.0).sum()
                        u1_N+=100000
                    u1_rate=u1_hits/u1_N; u1_sigma=np.sqrt(u1_hits)/u1_N
                    print(f"    (i) P(U1) singleton f=1: {u1_rate:.6e}±{u1_sigma:.6e} ({u1_hits}/{u1_N})")

                    # (ii) Q2^U1 pair
                    print(f"    (ii) Q2^U1 pair...")
                    q2u1_claims = ["Q2", f"U1_{u1_mode}"]
                    q2u1_ok = False
                    for f_pair in INFLATION_FACTORS:
                        nv_q2,_=nonvacuous("Q2",f_pair); nv_u1,_=nonvacuous(f"U1_{u1_mode}",f_pair)
                        if not nv_q2 or not nv_u1:
                            print(f"      Factor {f_pair}: VACUOUS"); continue
                        res_pair, ok_pair = validate_stage_optimized(
                            condition, u1_mode, "+U1_pair", lep_claims, q2u1_claims,
                            lepton_factor, calibration_seed+66666, N_max
                        )
                        if res_pair.get("eligible") and res_pair.get("bf_hits",0)>=100:
                            q2u1_ok = ok_pair
                            print(f"      Factor {f_pair}: {'✓' if ok_pair else '✗'} "
                                  f"{res_pair.get('deviation_sigma',float('inf')):.2f}σ")
                            break

                    comp_ok = bool(u1_hits>=100 and q2u1_ok)
                    validation_record[stage_name] = {
                        "method":"compositional", "eligible":True,
                        "u1_singleton":{"rate":float(u1_rate),"hits":int(u1_hits),"N":int(u1_N)},
                        "q2u1_validated":bool(q2u1_ok),
                        "validated_by_composition":comp_ok, "within_2sigma":comp_ok,
                    }
                    if not comp_ok: all_validated = False
                    print(f"    VALIDATED-BY-COMPOSITION: {'✓' if comp_ok else '✗'}")
            else:
                result, ok = validate_stage_optimized(
                    condition, u1_mode, stage_name, lep_claims, new_claims,
                    lepton_factor, calibration_seed, N_max
                )
                validation_record[stage_name] = result
                if not ok: all_validated = False

    # ── Save ──
    outpath = os.path.join(args.outdir, f"tier2_validation_{args.cell}_v0.5_optimized.json")
    output = {
        "engine_id":"amb", "spec_version":"v0.5",
        "cell":args.cell, "condition":condition, "u1_mode":u1_mode,
        "calibration_seed":calibration_seed, "N_max":N_max,
        "lepton_factor":lepton_factor,
        "all_stages_validated":bool(all_validated),
        "validation_record":validation_record,
    }
    with open(outpath,"w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {outpath}")

    if all_validated:
        print("\n✓ ALL STAGES VALIDATED — Tier-2 estimates are TIER-2 (VALIDATED)")
    else:
        failed=[s for s,r in validation_record.items() if not r.get("within_2sigma",False)]
        print(f"\n✗ VALIDATION FAILED — stages: {failed}")
        print("Tier-2 estimates UNVALIDATED. Tier-1 bounds remain the headline.")

    return all_validated, validation_record

if __name__ == "__main__":
    main()
