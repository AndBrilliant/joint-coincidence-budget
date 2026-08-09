# Joint Coincidence Budget (P3+P4)

**ENGINE_ID:** amb  
**Spec:** v0.3 (audit-response revision, `specs/SPEC_V03.md`)  
**Status:** Tier-1 production, accepted-N semantics, 20-target U1 menu, |log(r/r0)| criterion

## Reproduction

```bash
git clone git@github.com:AndBrilliant/joint-coincidence-budget.git
cd joint-coincidence-budget
pip install -r requirements.txt
python3 scripts/joint_engine_v0.3.py --mode full --outdir results/amb-v0.3
```

Gates-only (no production — ~3 seconds):
```bash
python3 scripts/joint_engine_v0.3.py --mode gates
```

## What this computes

The joint probability that a world drawn from a null distribution over the
Standard Model flavor parameters simultaneously satisfies all observed
coincidences:

- **L1:** Lepton Koide distance ≤ 3.3049×10⁻⁶
- **L2:** |log(m₂/m₁ / 206.7703)| ≤ 1.00×10⁻⁵
- **L3:** |log(m₃/m₂ / 16.8180)| ≤ 2.10×10⁻⁵
- **Q1:** |log(mₛ²/(μ*m_d))| ≤ 3.00×10⁻³  (anchored on lepton draw)
- **Q2:** |log(mᵤ²/(m_d·2mₑ))| ≤ 1.18×10⁻²  (anchored on lepton draw)
- **U1:** |9·Q_U − 9·(p/q)| ≤ 1.1414×10⁻² for any irreducible p/q with p,q≤9, 1/3 ≤ p/q ≤ 1 (20 targets)

Two null conditions:
- **T0:** Three independent log-uniform lepton masses on [0.3, 2000] MeV
- **T1:** Koide-sheet measure, log-uniform in (m₃, m₁/m₃), m₂ from Koide equation

## v0.3 key changes (from v0.2)

1. **Accepted-N semantics:** N_eff = N accepted worlds. L1@T1 = 1.000000 by construction.
2. **|log(r/r₀)| ≤ ε** for L2/L3 (was |r/r₀−1| ≤ ε).
3. **20-target U1 menu** (1/3 included; v0.2 used 19).
4. **True truncated normal** (scipy.stats.truncnorm) for logN; clip variant renamed 'censored'.
5. **Alt-sheet robustness:** T1 variant parameterized as log-uniform in (m₂, m₁/m₂).

## Gates

| Gate | Description | Status |
|------|-------------|--------|
| G1 | Golden regression: 69/10⁷ hits (T0, seed 20260726) | — |
| G2 | T1 L1 singleton = 1.000000 (accepted-N denominator) | — |
| G3 | Support gates (r_obs ∈ [1e-5, 1e-1], m₃/m₁ > 67.9) | — |

Gates stop on first mismatch; never tuned.

## Directory

| Path | Contents |
|------|----------|
| `specs/SPEC_V03.md` | Frozen v0.3 specification |
| `scripts/joint_engine_v0.3.py` | Production engine |
| `scripts/golden_regression.py` | G1 gate (v0.2 legacy, preserved) |
| `inputs_frozen.json` | Frozen claim table, tolerances, observed values |
| `ASSUMPTIONS.md` | Conservative readings, implementation decisions |
| `PREREGISTRATION.md` | Study preregistration |
| `REPORT.md` | Results report (v0.5) |
| `VOID.md` | Void declarations |
| `results/amb-v0.3/` | v0.3 production output |
| `results/amb-20260811-v0.5/` | v0.5 results (archived) |
| `results/amb-20260811-v0.2/` | v0.2 results (archived, T0 valid, T1 pre-audit) |

## License

All rights reserved. Contact ab@ad-research.org.
