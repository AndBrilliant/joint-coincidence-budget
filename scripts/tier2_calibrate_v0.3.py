#!/usr/bin/env python3
"""
Tier-2 Adaptive Calibration — v0.3 (self-contained)
Runs adaptive per-stage inflation validation.
"""
import numpy as np
from scipy.stats import beta as beta_dist
import json, os, sys, time

# ── All needed functions copied from joint_engine_v0.3.py ──

ang = 2 * np.pi * np.arange(3) / 3
cos_ang = np.cos(ang)
sin_ang = np.sin(ang)

def kdist(m):
    m = np.asarray(m)
    out = None
    for v in (np.sqrt(m), 1.0 / np.sqrt(m)):
        A = v.mean(axis=-1)
        X = (2.0 / 3.0) * (v * cos_ang).sum(axis=-1)
        Y = -(2.0 / 3.0) * (v * sin_ang).sum(axis=-1)
        d = np.abs(np.hypot(X, Y) / (np.sqrt(2.0) * A) - 1.0)
        out = d if out is None else np.minimum(out, d)
    return out

def Q_U(v):
    v = np.asarray(v)
    return np.sum(v, axis=-1) / np.sum(np.sqrt(v), axis=-1) ** 2

def clopper_pearson(k, n, alpha=0.05):
    if k <= 0: lower = 0.0
    else: lower = beta_dist.ppf(alpha / 2.0, k, n - k + 1)
    if k >= n: upper = 1.0
    else: upper = beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(lower), float(upper)

# Constants
LEPTONS_OBS = np.array([0.51099895, 105.6583755, 1776.93])
L1_TOL = 3.3049e-6
L2_TARGET = 206.7703
L2_TOL = 1.00e-5
L3_TARGET = 16.8180
L3_TOL = 2.10e-5
B1 = 3.00e-3
B2 = 1.18e-2
U1_TOL = 1.1414e-2

U1_MENU = [(1,1),(1,2),(2,3),(3,4),(2,5),(3,5),(4,5),(5,6),
           (3,7),(4,7),(5,7),(6,7),(3,8),(5,8),(7,8),(4,9),(5,9),(7,9),(8,9)]
U1_MENU_TARGETS = np.array([9.0 * p / q for (p, q) in U1_MENU], dtype=np.float64)

LEP_LO, LEP_HI = 0.3, 2000.0
QUARK_LO, QUARK_HI = 0.5, 2e5
LEP_LOG_LO, LEP_LOG_HI = np.log(LEP_LO), np.log(LEP_HI)
LEP_LOG_V = LEP_LOG_HI - LEP_LOG_LO
QUARK_LOG_LO, QUARK_LOG_HI = np.log(QUARK_LO), np.log(QUARK_HI)
QUARK_LOG_V = QUARK_LOG_HI - QUARK_LOG_LO
HIERARCHY_MIN = (4.0 + np.sqrt(18.0)) ** 2
INFLATION_FACTORS = [1, 3, 10, 30, 100, 300, 1000]

def draw_mass_triple(rng, batch_size, lo, hi, prior, sort=True):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 3)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 3)))
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 3))
    if sort: x.sort(axis=1)
    return x

def draw_mass_pair(rng, batch_size, lo, hi, prior):
    if prior == "logU":
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(batch_size, 2)))
    elif prior == "logN":
        mid = np.exp((np.log(lo) + np.log(hi)) / 2.0)
        x = np.exp(rng.normal(np.log(mid), 1.5 * np.log(10), size=(batch_size, 2)))
        x = np.clip(x, lo, hi)
    elif prior == "linU":
        x = rng.uniform(lo, hi, size=(batch_size, 2))
    return x

def sample_t1_koide(rng, batch_size):
    m3 = np.exp(rng.uniform(np.log(LEP_LO), np.log(LEP_HI), size=batch_size))
    r = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), size=batch_size))
    m1 = r * m3
    s1, s3 = np.sqrt(m1), np.sqrt(m3)
    b = -4.0 * (s1 + s3)
    c_coeff = s1**2 + s3**2 - 4.0 * s1 * s3
    disc = np.maximum(b**2 - 4.0 * c_coeff, 0.0)
    s2 = (-b - np.sqrt(disc)) / 2.0
    keep = (disc >= 0) & (s2 > 0) & (m1 < s2**2) & (s2**2 < m3) & ((m3 / m1) > HIERARCHY_MIN)
    m2 = s2[keep] ** 2
    return np.column_stack([m1[keep], m2, m3[keep]]), batch_size

# Inflated checks
def check_L1_inflated(lep, f):
    return kdist(lep) <= L1_TOL * f

def check_L2_inflated(lep, f):
    ratio = lep[:, 1] / lep[:, 0]
    return np.abs(ratio / L2_TARGET - 1.0) <= L2_TOL * f

def check_L3_inflated(lep, f):
    ratio = lep[:, 2] / lep[:, 1]
    return np.abs(ratio / L3_TARGET - 1.0) <= L3_TOL * f

def check_Q1_inflated(lq, lep, f):
    mu, md, ms = lq[:, 0], lq[:, 1], lq[:, 2]
    mu_star = lep.sum(axis=1)
    return np.abs(np.log(ms*ms/(mu_star*md))) <= B1 * f

def check_Q2_inflated(lq, lep, f):
    mu, md = lq[:, 0], lq[:, 1]
    twome = 2.0 * lep.min(axis=1)
    return np.abs(np.log(mu*mu/(md*twome))) <= B2 * f

def check_U1_fixed_inflated(lq, us, f):
    mu, mc, mt = lq[:, 0], us[:, 0], us[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0/t)
    return np.minimum(np.abs(9*qd-8), np.abs(9*qi-8)) <= U1_TOL*f

def check_U1_menu_inflated(lq, us, f):
    mu, mc, mt = lq[:, 0], us[:, 0], us[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0/t)
    hit = np.zeros(len(mu), dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9*qd-tgt) <= U1_TOL*f)
        hit |= (np.abs(9*qi-tgt) <= U1_TOL*f)
    return hit

# Non-inflated checks
def check_Q1(lq, lep):
    mu, md, ms = lq[:, 0], lq[:, 1], lq[:, 2]
    return np.abs(np.log(ms*ms/(lep.sum(axis=1)*md))) <= B1

def check_Q2(lq, lep):
    mu, md = lq[:, 0], lq[:, 1]
    return np.abs(np.log(mu*mu/(md*2.0*lep.min(axis=1)))) <= B2

def check_U1_fixed(lq, us):
    mu, mc, mt = lq[:, 0], us[:, 0], us[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0/t)
    return np.minimum(np.abs(9*qd-8), np.abs(9*qi-8)) <= U1_TOL

def check_U1_menu(lq, us):
    mu, mc, mt = lq[:, 0], us[:, 0], us[:, 1]
    t = np.column_stack([mu, mc, mt])
    qd, qi = Q_U(t), Q_U(1.0/t)
    hit = np.zeros(len(mu), dtype=bool)
    for tgt in U1_MENU_TARGETS:
        hit |= (np.abs(9*qd-tgt) <= U1_TOL)
        hit |= (np.abs(9*qi-tgt) <= U1_TOL)
    return hit

# ── Non-vacuous checks ──
def check_nonvacuous_lepton(claim_id, factor):
    V = LEP_LOG_V
    if claim_id == "L1":
        rng = np.random.default_rng(271828)
        N = 500000
        x = rng.uniform(LEP_LOG_LO, LEP_LOG_HI, size=(N, 3))
        x.sort(axis=1)
        m = np.exp(x)
        frac = check_L1_inflated(m, factor).mean()
        return frac < 0.5, float(frac)
    elif claim_id in ("L2", "L3"):
        target = np.log(L2_TARGET) if claim_id == "L2" else np.log(L3_TARGET)
        tol = L2_TOL if claim_id == "L2" else L3_TOL
        hw = tol * factor
        lo, hi = max(0, target-hw), min(V, target+hw)
        if lo >= hi: return True, 0.0
        frac = (2*(hi-lo)/V) - ((hi/V)**2 - (lo/V)**2)  # triangular CDF
        return frac < 0.5, float(max(0, frac))
    return True, 0.0

def check_nonvacuous_quark(claim_id, factor):
    V_q = QUARK_LOG_V
    if claim_id == "Q1":
        frac = 2.0 * B1 * factor / V_q
        return frac < 0.5, float(min(frac, 1.0))
    elif claim_id == "Q2":
        frac = 2.0 * B2 * factor / V_q
        return frac < 0.5, float(min(frac, 1.0))
    elif claim_id in ("U1_fixed", "U1_menu"):
        rng = np.random.default_rng(271828)
        N = 200000
        lq = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 3)))
        lq.sort(axis=1)
        us = np.exp(rng.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N, 2)))
        if claim_id == "U1_fixed":
            frac = check_U1_fixed_inflated(lq, us, factor).mean()
        else:
            frac = check_U1_menu_inflated(lq, us, factor).mean()
        return frac < 0.5, float(frac)
    return True, 0.0

# ── Tier-2 lepton block (copied from main engine) ──
def tier2_lepton_block_t0():
    u0, v0 = np.log(L2_TARGET), np.log(L3_TARGET)
    V = LEP_LOG_V
    du, dv = L2_TOL, L3_TOL
    s_width = V - u0 - v0
    area_uv = (2 * du) * (2 * dv)
    N_t = 10000
    rng = np.random.default_rng(271828)
    u_t = u0 + rng.uniform(-du, du, N_t)
    v_t = v0 + rng.uniform(-dv, dv, N_t)
    m1_t = np.ones(N_t) * LEPTONS_OBS[0]
    m2_t = m1_t * np.exp(u_t)
    m3_t = m2_t * np.exp(v_t)
    l_t = np.column_stack([m1_t, m2_t, m3_t])
    f_l1 = float((kdist(l_t) <= L1_TOL).mean())
    p_lep = 6.0 / (V**3) * s_width * area_uv * f_l1
    p_l2l3 = 6.0 / (V**3) * s_width * area_uv
    print(f"  [T2 T0 lepton] P(L1∧L2∧L3)={p_lep:.6e}, P(L2∧L3)={p_l2l3:.6e}, L1_frac={f_l1:.4f}")
    return p_lep, p_l2l3, f_l1

def tier2_lepton_block_t1():
    V_m3 = LEP_LOG_V
    V_r = np.log(1e-1 / 1e-5)
    r_test = np.logspace(np.log10(1e-5), np.log10(1e-1), 2000000)
    x_test = np.sqrt(r_test)
    s2s3 = 2*(x_test+1) - np.sqrt(np.maximum(0, 3*(x_test**2 + 4*x_test + 1)))
    m2m1 = s2s3**2 / x_test**2
    m3m2 = 1.0 / s2s3**2
    l2 = np.abs(m2m1/L2_TARGET - 1) <= L2_TOL
    l3 = np.abs(m3m2/L3_TARGET - 1) <= L3_TOL
    both = l2 & l3
    if both.any():
        rb = r_test[both]
        dlnr = np.log(rb.max()/rb.min())
        p_l2l3 = dlnr / V_r
        print(f"  [T2 T1 lepton] P(L2∧L3)={p_l2l3:.6e} (Δln(r)={dlnr:.6e}, V_r={V_r:.4f})")
    else:
        p_l2l3 = 0.0
        print(f"  [T2 T1 lepton] NO r values satisfy L2∧L3!")
    return p_l2l3, {"V_r": float(V_r)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN: Smart calibration
# ═══════════════════════════════════════════════════════════════════════

def run_calibration(prior, condition, u1_mode, calibration_seed=271828, N_max=1_000_000_000):
    batch_size = 1_000_000

    if condition == "T0":
        stage_defs = [
            ("L1_L2_L3", ["L1", "L2", "L3"]),
            ("+Q1", ["L1", "L2", "L3", "Q1"]),
            ("+Q2", ["L1", "L2", "L3", "Q1", "Q2"]),
            ("+U1", ["L1", "L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]),
        ]
    else:
        stage_defs = [
            ("L2_L3", ["L2", "L3"]),
            ("+Q1", ["L2", "L3", "Q1"]),
            ("+Q2", ["L2", "L3", "Q1", "Q2"]),
            ("+U1", ["L2", "L3", "Q1", "Q2", f"U1_{u1_mode}"]),
        ]

    calibration_record = {}
    all_validated = True

    for stage_name, claim_list in stage_defs:
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}  ({', '.join(claim_list)})")
        print(f"{'='*60}")

        stage_result = {"claims": claim_list, "factors_tested": {}}
        found = False

        for factor in INFLATION_FACTORS:
            if found:
                break

            # Non-vacuous check
            vacuous = False
            for claim in claim_list:
                cb = claim.replace("U1_fixed","X").replace("U1_menu","X")
                if cb in ("L1","L2","L3"):
                    nv, fr = check_nonvacuous_lepton(claim, factor)
                elif cb in ("Q1","Q2","X"):
                    c_map = {"Q1":"Q1","Q2":"Q2","X":"U1_fixed" if "fixed" in claim else "U1_menu"}
                    nv, fr = check_nonvacuous_quark(c_map.get(cb, cb), factor)
                else:
                    nv, fr = True, 0.0
                if not nv:
                    print(f"  factor={factor}: VACUOUS for {claim} ({fr*100:.1f}%)")
                    vacuous = True
                    break

            if vacuous:
                stage_result["factors_tested"][str(factor)] = {"vacuous": True}
                continue

            # Estimate rate
            N_est = 3_000_000
            rng_est = np.random.default_rng(calibration_seed + hash(stage_name) % 10000)
            lep_b = np.tile(LEPTONS_OBS, (N_est, 1))
            lq = np.exp(rng_est.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_est, 3)))
            lq.sort(axis=1)
            us = np.exp(rng_est.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_est, 2)))

            surv = np.ones(N_est, dtype=bool)
            for claim in claim_list:
                check_map = {
                    "L1": lambda: check_L1_inflated(lep_b, factor),
                    "L2": lambda: check_L2_inflated(lep_b, factor),
                    "L3": lambda: check_L3_inflated(lep_b, factor),
                    "Q1": lambda: check_Q1_inflated(lq, lep_b, factor),
                    "Q2": lambda: check_Q2_inflated(lq, lep_b, factor),
                    "U1_fixed": lambda: check_U1_fixed_inflated(lq, us, factor),
                    "U1_menu": lambda: check_U1_menu_inflated(lq, us, factor),
                }
                m = check_map[claim]()
                si = np.where(surv)[0]
                surv[si[~m[si]]] = False

            est_rate = float(surv.mean())
            # Scale up for T0/T1 (MC used only quark side, need lepton factor)
            # For T0: need lepton block probability too
            if condition == "T0":
                V = LEP_LOG_V
                u0, v0 = np.log(L2_TARGET), np.log(L3_TARGET)
                area_uv = (2*L2_TOL*factor)*(2*L3_TOL*factor)
                f_l1 = float((kdist(lep_b[:50000]) <= L1_TOL*factor).mean())
                p_lep_est = 6.0/(V**3)*(V-u0-v0)*area_uv*f_l1
                joint_rate = p_lep_est * est_rate
            else:
                # T1: lepton factor for L2∧L3 already in p_lep
                # But our MC used fixed observed leptons which auto-satisfy L2∧L3
                # So est_rate is P(Q1∧Q2∧U1 | L2∧L3 satisfied)
                # Need to multiply by P(L2∧L3) from T1 sheet
                r_t = np.logspace(np.log10(1e-5), np.log10(1e-1), 2000000)
                x_t = np.sqrt(r_t)
                s2t = 2*(x_t+1)-np.sqrt(np.maximum(0,3*(x_t**2+4*x_t+1)))
                m2m1t = s2t**2/x_t**2
                m3m2t = 1.0/s2t**2
                both_t = (np.abs(m2m1t/L2_TARGET-1)<=L2_TOL*factor) & (np.abs(m3m2t/L3_TARGET-1)<=L3_TOL*factor)
                if both_t.any():
                    rb_t = r_t[both_t]
                    p_lep_est = np.log(rb_t.max()/rb_t.min())/np.log(1e-1/1e-5)
                else:
                    p_lep_est = 0.0
                joint_rate = p_lep_est * est_rate

            exp_hits = joint_rate * N_max
            print(f"  factor={factor}: joint_rate={joint_rate:.3e}, exp_hits@1e9={exp_hits:.1f}", flush=True)

            if exp_hits < 30:
                stage_result["factors_tested"][str(factor)] = {
                    "vacuous": False, "skipped": True,
                    "est_rate": float(joint_rate),
                }
                continue

            # ── Brute force ──
            rng_bf = np.random.default_rng(calibration_seed)
            total_eff, total_hits, n_b = 0, 0, 0
            t0 = time.time()

            while total_eff < N_max and total_hits < 100:
                n_b += 1
                if condition == "T0":
                    lep = draw_mass_triple(rng_bf, batch_size, LEP_LO, LEP_HI, prior, sort=True)
                    n_acc = batch_size
                    total_eff += batch_size
                else:
                    lep, attempted = sample_t1_koide(rng_bf, batch_size)
                    total_eff += attempted
                    n_acc = len(lep)
                    if n_acc == 0: continue

                lq = draw_mass_triple(rng_bf, n_acc, QUARK_LO, QUARK_HI, prior, sort=True)
                us = draw_mass_pair(rng_bf, n_acc, QUARK_LO, QUARK_HI, prior)
                survivors = np.ones(n_acc, dtype=bool)

                for claim in claim_list:
                    cmap2 = {
                        "L1": lambda: check_L1_inflated(lep, factor),
                        "L2": lambda: check_L2_inflated(lep, factor),
                        "L3": lambda: check_L3_inflated(lep, factor),
                        "Q1": lambda: check_Q1_inflated(lq, lep, factor),
                        "Q2": lambda: check_Q2_inflated(lq, lep, factor),
                        "U1_fixed": lambda: check_U1_fixed_inflated(lq, us, factor),
                        "U1_menu": lambda: check_U1_menu_inflated(lq, us, factor),
                    }
                    mask = cmap2[claim]()
                    si = np.where(survivors)[0]
                    survivors[si[~mask[si]]] = False

                total_hits += survivors.sum()
                if n_b % 50 == 0:
                    print(f"    [{stage_name}/x{factor}] N={total_eff:,}, hits={total_hits}", flush=True)

            elapsed = time.time() - t0
            bf_rate = total_hits / total_eff if total_eff > 0 else 0.0
            print(f"    [{stage_name}/x{factor}] DONE: hits={total_hits}, N={total_eff:,}, rate={bf_rate:.3e}, {elapsed:.0f}s")

            stage_result["factors_tested"][str(factor)] = {
                "vacuous": False,
                "hits": int(total_hits), "N_eff": int(total_eff),
                "rate": float(bf_rate),
            }

            if total_hits >= 100:
                stage_result["best_factor"] = factor
                stage_result["best_bf_hits"] = int(total_hits)
                stage_result["best_bf_n"] = int(total_eff)
                found = True

                # Validation
                bf_sigma = np.sqrt(total_hits) / total_eff
                # Tier-2 estimate
                if condition == "T0":
                    V = LEP_LOG_V; u0v, v0v = np.log(L2_TARGET), np.log(L3_TARGET)
                    area_f = (2*L2_TOL*factor)*(2*L3_TOL*factor)
                    f_l1_f = check_L1_inflated(lep_b[:50000], factor).mean()
                    p_lep_f = 6.0/(V**3)*(V-u0v-v0v)*area_f*f_l1_f
                else:
                    rtv = np.logspace(np.log10(1e-5), np.log10(1e-1), 2000000)
                    xtv = np.sqrt(rtv)
                    s2tv = 2*(xtv+1)-np.sqrt(np.maximum(0,3*(xtv**2+4*xtv+1)))
                    m2m1tv = s2tv**2/xtv**2; m3m2tv=1.0/s2tv**2
                    btv = (np.abs(m2m1tv/L2_TARGET-1)<=L2_TOL*factor)&(np.abs(m3m2tv/L3_TARGET-1)<=L3_TOL*factor)
                    p_lep_f = np.log(rtv[btv].max()/rtv[btv].min())/np.log(1e-1/1e-5) if btv.any() else 0.0

                p_quark_f = est_rate  # from earlier MC estimate
                t2_est = p_lep_f * p_quark_f

                within = abs(t2_est - bf_rate) <= 2.0 * bf_sigma
                stage_result["t2_est"] = float(t2_est)
                stage_result["bf_rate"] = float(bf_rate)
                stage_result["bf_sigma"] = float(bf_sigma)
                stage_result["within_2sigma"] = bool(within)
                print(f"  Validation: T2={t2_est:.3e}, BF={bf_rate:.3e}±{bf_sigma:.3e}, 2σ={within}")
                if not within:
                    all_validated = False

        if not found:
            print(f"  Stage {stage_name}: INELIGIBLE — no factor reached 100 hits")
            stage_result["eligible"] = False
            all_validated = False
        else:
            stage_result["eligible"] = True

        calibration_record[stage_name] = stage_result

    # ── Tier-2 point estimate at factor 1 ──
    if condition == "T0":
        p_lep, _, _ = tier2_lepton_block_t0()
    else:
        p_lep, _ = tier2_lepton_block_t1()

    N_q = 10_000_000
    rng_q = np.random.default_rng(calibration_seed + 99999)
    lb = np.tile(LEPTONS_OBS, (N_q, 1))
    lq = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_q, 3)))
    lq.sort(axis=1)
    us = np.exp(rng_q.uniform(QUARK_LOG_LO, QUARK_LOG_HI, size=(N_q, 2)))

    q1 = check_Q1(lq, lb)
    q2 = check_Q2(lq, lb)
    if u1_mode == "fixed":
        u1 = check_U1_fixed(lq, us)
    else:
        u1 = check_U1_menu(lq, us)

    q1q2 = q1 & q2
    joint = q1q2 & u1
    p_quark = float(joint.mean())
    p_joint = p_lep * p_quark

    # CP95 for quark block
    k_q = int(joint.sum())
    lo_q, hi_q = clopper_pearson(k_q, N_q)

    print(f"\n{'='*60}")
    print(f"Tier-2 Point Estimate (factor 1):")
    print(f"  P(lepton block)  = {p_lep:.6e}")
    print(f"  P(quark block)   = {p_quark:.6e} [{lo_q:.6e}, {hi_q:.6e}] (CP95, k={k_q}/{N_q})")
    print(f"  P(joint)         = {p_joint:.6e}")
    print(f"  All validated:   = {all_validated}")
    print(f"{'='*60}")

    return calibration_record, all_validated, {
        "p_lepton_block": float(p_lep),
        "p_quark_block": float(p_quark),
        "p_quark_hits": int(k_q),
        "p_quark_N": N_q,
        "p_quark_cp95": [float(lo_q), float(hi_q)],
        "p_joint": float(p_joint),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="logU")
    p.add_argument("--condition", default="T0")
    p.add_argument("--u1-mode", default="fixed")
    p.add_argument("--outdir", default="results/calibration-v0.3")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Tier-2 Calibration: {args.condition}/{args.u1_mode}/{args.prior}")
    calib, validated, estimates = run_calibration(
        args.prior, args.condition, args.u1_mode
    )

    out = {
        "engine_id": "amb",
        "spec_version": "v0.3",
        "condition": args.condition,
        "u1_mode": args.u1_mode,
        "prior": args.prior,
        "calibration_seed": 271828,
        "calibration_record": calib,
        "tier2_validated": validated,
        "tier2_estimates": estimates,
    }
    outpath = os.path.join(args.outdir,
        f"calibration_{args.condition}_{args.u1_mode}_{args.prior}.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}")
