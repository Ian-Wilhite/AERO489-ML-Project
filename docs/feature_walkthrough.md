# Feature Walkthrough — AERO 489 Wing Simulation Dataset

**Dataset:** 476 ABAQUS simulations | **Target:** `g_limit` — maximum sustained g-load before structural failure

---

## Target Variable

### `g_limit`
**What it is:** The maximum sustained g-load (multiples of Earth gravity) the wing can carry before the simulation reaches structural failure. Computed as `2 × RF_failure / (m_aircraft × g)`, where the factor of 2 accounts for one wing carrying half the aircraft's inertial load.

**Units:** dimensionless (g)

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 0.046 | 0.232 | 1.022 | 1.674 | 1.679 |

**Interpretation:** The median of ~1 g is physically meaningful — a wing supporting normal level flight sustains exactly 1 g. The sharp upper bound near 1.68 g across Q3 and the max suggests the undamaged wing has a hard structural ceiling around that value. The long lower tail (min 0.046 g, Q1 at 0.23 g) represents severely damaged configurations that fail under almost no maneuvering load. The IQR spanning 0.23–1.67 g gives the model a rich dynamic range to learn from.

---

## Simulation Metadata

### `n_steps`
**What it is:** Number of converged ABAQUS load increments in the simulation. ABAQUS uses an incremental-iterative solver; more increments means the solver stepped through the loading history in finer resolution before reaching failure.

**Units:** integer (count) | **r with g_limit:** +0.68

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 2 | 10 | 18 | 18 | 29 |

**Interpretation:** The median and Q3 are both 18, meaning the majority of simulations ran through the standard 18-step load ramp. Q1 = 10 indicates a quarter of simulations hit a failure condition early, cutting the ramp short — these correspond to heavily damaged wings. The minimum of 2 steps is an extreme case: the wing failed almost immediately on first load application. The moderate correlation (r = +0.68) confirms that longer ramps track with higher load capacity.

### `RF_failure`
**What it is:** Total reaction force at the wing root (sum of all nodal constraint forces) at the last converged load step — the force the wing was carrying when ABAQUS could no longer find equilibrium.

**Units:** Newtons (N) | **r with g_limit:** +1.00

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 3,740 | 18,757 | 82,743 | 135,452 | 135,907 |

**Interpretation:** This is the direct physical precursor to `g_limit` (r = +1.00 by construction, since g_limit is derived from it). The nearly identical Q3 and max (~135,500 N) correspond to the undamaged wing's structural limit. The median of ~82,700 N maps to 1.02 g. The dramatic drop to Q1 = 18,757 N reflects the damaged-wing population. As a predictor it is informationally perfect but operationally useless — you cannot measure failure force before the wing fails.

---

## Feature Group A — Tip Deflection

### `tip_deflection_at_failure`
**What it is:** The spanwise tip displacement of the wing at the final load step — how far the wingtip has moved out-of-plane when the simulation terminates.

**Units:** meters (m) | **r with g_limit:** +0.87

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 0.0136 | 0.0689 | 0.293 | 0.358 | 0.711 |

**Interpretation:** The median deflection of ~29 cm at failure is substantial. The tight Q3 (35.8 cm) near the undamaged-wing cluster and the large upper tail (max 71 cm) indicate that a few very compliant or heavily loaded configurations undergo extreme bending before failure. The minimum of 1.4 cm comes from severely damaged wings that collapse at almost zero load. Strong correlation (r = +0.87): more deflection generally means more load was applied before failure.

### `tip_deflection_slope`
**What it is:** The slope of a linear fit to tip deflection vs. reaction force across all load steps — a measure of the wing's structural compliance (inverse stiffness).

**Units:** m/N | **r with g_limit:** −0.26

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 2.62×10⁻⁶ | 2.66×10⁻⁶ | 2.81×10⁻⁶ | 3.38×10⁻⁶ | 6.57×10⁻⁵ |

**Interpretation:** The IQR is extremely narrow (2.66–3.38 nm/N), meaning the vast majority of wings — regardless of damage state — deflect at nearly the same rate per Newton applied. This makes physical sense: damage changes the failure load, not necessarily the early-regime stiffness. The extreme outlier at 6.57×10⁻⁵ m/N (25× the median) is a wing whose damage has catastrophically reduced stiffness. The weak negative correlation (r = −0.26) reflects that a more compliant wing tends to fail at a lower load.

### `tip_per_g_at_failure`
**What it is:** Tip deflection normalized by the g-load at the moment of failure: `tip_deflection / g_loading_at_failure`. Captures how much the wingtip has moved per unit of aerodynamic loading at the instant of failure.

**Units:** m/g | **r with g_limit:** −0.26

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 0.213 | 0.216 | 0.228 | 0.273 | 5.38 |

**Interpretation:** The extremely tight cluster between 0.21 and 0.27 m/g for 75% of simulations reveals a near-constant structural ratio — the wing deflects roughly 22 cm per g regardless of damage state, up to the point of failure. The outlier at 5.38 m/g is a severely softened wing reaching failure at a very low g but with disproportionate deflection. The weak negative correlation (r = −0.26) mirrors `tip_deflection_slope` since the two are algebraically related.

---

## Feature Group B — Average Strain

### `avg_strain_at_failure`
**What it is:** The arithmetic mean of all 24 strain gauge readings at the final load step — a single-number summary of the overall strain field in the wing at failure.

**Units:** dimensionless (m/m; multiply by 10⁶ to read in microstrain µε) | **r with g_limit:** +0.95

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 9.4×10⁻⁶ | 9.3×10⁻⁵ | 3.7×10⁻⁴ | 6.1×10⁻⁴ | 9.2×10⁻⁴ |

**Interpretation:** Equivalently: 9.4 µε (min) through 366 µε (median) to 916 µε (max). The two-order-of-magnitude span directly tracks with how much load the wing absorbed. Stronger correlation with `g_limit` (r = +0.95) than any engineered feature except `RF_failure` itself — the average strain field is a near-sufficient statistic for how hard the wing was working at failure.

### `avg_strain_slope`
**What it is:** Slope of a linear fit to mean strain vs. reaction force — how rapidly the average strain field grows per Newton of applied load.

**Units:** (m/m)/N | **r with g_limit:** −0.14

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 1.57×10⁻⁹ | 4.15×10⁻⁹ | 4.75×10⁻⁹ | 5.44×10⁻⁹ | 1.92×10⁻⁸ |

**Interpretation:** Like `tip_deflection_slope`, the IQR is very narrow — most wings accumulate strain at nearly the same rate per Newton. The upper tail (max = 4× the median) catches the most damaged wings where the same force produces much higher strain. Weak negative correlation (r = −0.14): a wing that strains rapidly per Newton tends to fail sooner, but this slope alone carries little predictive signal beyond what the at-failure value already captures.

---

## Feature Group C — Strain Energy

### `strain_energy_at_failure`
**What it is:** A proxy for elastic strain energy stored in the wing at failure: `0.5 × E × Σεᵢ²` summed across all 24 gauges. The true strain energy requires multiplying by node volumes (not available here), so this is dimensionally a stress rather than a true energy.

**Units:** Pa (proxy; multiply by average node volume in m³ to recover Joules) | **r with g_limit:** +0.88

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 166 | 12,630 | 208,458 | 549,258 | 1,407,700 |

**Interpretation:** The four-order-of-magnitude range (166 Pa to 1.4 MPa proxy) dwarfs any other feature in spread. The mean (294,100) substantially exceeds the median (208,458), confirming a right-skewed distribution — a few high-g undamaged cases dominate the upper tail. Strong correlation (r = +0.88): more energy stored = more work done on the structure = higher g achieved. As a quadratic function of strain, it amplifies differences that `avg_strain_at_failure` captures linearly.

### `strain_energy_slope`
**What it is:** Slope of the strain energy proxy vs. reaction force — rate of elastic energy accumulation per Newton.

**Units:** Pa/N (proxy) | **r with g_limit:** +0.75

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 0.052 | 0.689 | 2.48 | 4.45 | 20.62 |

**Interpretation:** The IQR (0.69–4.45) is wide relative to the median, reflecting genuine variation in how fast different damage configurations store energy. The max of 20.6 is ~8× the median — a heavily damaged wing that stores disproportionate strain energy per Newton due to stress concentration. Moderate correlation (r = +0.75): unlike the deflection and strain slopes, this one adds real signal because it captures both stiffness degradation and strain field geometry.

---

## Max Von Mises Stress

### `max_vm_stress_at_failure`
**What it is:** The maximum von Mises stress anywhere in the wing model at the final converged load step — the peak stress state the structure reached before the solver diverged.

**Units:** Pa (MPa when divided by 10⁶) | **r with g_limit:** −0.61

| Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|
| 239 MPa | 249 MPa | 252 MPa | 261 MPa | 430 MPa |

**Interpretation:** The tight IQR (249–261 MPa) with a long upper tail is distinctive. Most simulations terminate near the same peak stress (~252 MPa) regardless of how much total load was applied — this is the material's effective failure threshold. Wings that reach 430 MPa did so with stress concentration: the damage created a hot-spot that hit the limit before the rest of the structure was loaded. The negative correlation (r = −0.61) is physically intuitive — if the peak stress is elevated at a given load level, the wing is less efficient and will fail at a lower g.

---

## Individual Strain Gauges (×24)

**Nodes:** 107, 151, 172, 180, 192, 201, 250, 257, 266, 281, 305, 327, 344, 368, 375, 407, 415, 447, 463, 472, 488, 502, 506, 510

**What they are:** Individual strain measurements at specific structural nodes at the final load step. Node numbering follows the ABAQUS mesh; higher node numbers do not necessarily correspond to more outboard positions.

**Units:** dimensionless (m/m) | **r with g_limit:** +0.45 to +0.63

### Representative statistics

| Stat | Typical range across all 24 gauges |
|---|---|
| Min | ~0 to slightly negative (local compression) |
| Q1 | 40–90 µε |
| Median | 130–370 µε |
| Q3 | 200–710 µε |
| Max | 1,750–3,640 µε |

### Per-gauge correlations with `g_limit`

| Node | r | Node | r | Node | r |
|---|---|---|---|---|---|
| 107 | +0.45 | 201 | +0.63 | 368 | +0.54 |
| 151 | +0.52 | 250 | +0.63 | 375 | +0.56 |
| 172 | +0.56 | 257 | +0.60 | 407 | +0.53 |
| 180 | +0.57 | 266 | +0.60 | 415 | +0.59 |
| 192 | +0.61 | 281 | +0.57 | 447 | +0.58 |
| — | — | 305 | +0.57 | 463 | +0.58 |
| — | — | 327 | +0.54 | 472 | +0.55 |
| — | — | 344 | +0.54 | 488 | +0.54 |
| — | — | — | — | 502 | +0.54 |
| — | — | — | — | 506 | +0.55 |
| — | — | — | — | 510 | +0.50 |

**Interpretation:** All 24 gauges share the same qualitative shape: near-zero minimums (lightly loaded or severely damaged wings), a moderate median cluster (130–370 µε), and sparse high outliers at 1,750–3,600 µε. Correlations range from r = +0.45 (Node 107) to r = +0.63 (Nodes 201, 250), all moderate positive. The variation in r across gauges reflects their spanwise and chordwise position — gauges closer to primary load paths tend to be better predictors. Nodes 172, 375, and 407 show slightly negative minimums, indicating that in some damage configurations local bending puts that node into compression even as the wing overall is in tension.

The individual gauges are weaker predictors than `avg_strain_at_failure` (r = +0.95) precisely because averaging all 24 suppresses single-gauge noise and captures the global strain state.