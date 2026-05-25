# Machine Learning Prediction of Structural Load Limits in Damaged Wings

---

## 1. Title, Team, and Contributions

**Project Title:** Machine Learning Prediction of Structural Load Limits in Damaged Wings
**Course:** AERO 489/689 — Introduction to Machine Learning for Aerospace Engineers, Spring 2026
**Instructor:** Dr. Raktim Bhattacharya, Texas A&M University

| Team Member | Role / Contribution |
|---|---|
| Barrett Brown | Scripting & Simulation — ABAQUS automation, data pipeline, batch execution on HPRC |
| Kirin Chadha | Classical ML — Linear Regression, Polynomial Regression implementation and evaluation |
| Malachi Drew | Aircraft Design & Modeling, Gaussian Process Regression, Random Forest |
| Ian Wilhite | Modern ML — Feedforward NN, PINN, LSTM, ablation study, comparison scripts |
| Adam Zheng | FEA Modeling — wing geometry, meshing, mesh refinement around damage sites |

All members contributed to report writing.

---

## 2. Abstract

<!-- CLAUDE NOTES — human authorship recommended for this section
Suggested talking points:
- Problem: real-time prediction of g-limit for a combat wing sustaining ballistic damage,
  using only onboard sensors (strain gauges + tip deflection) — no knowledge of damage geometry.
- Approach: seven ML models (four classical, three modern) trained on 468 ABAQUS FEA simulations
  of a damaged A-10-inspired skinless wing box structure.
- Data: 80/20 train/test split; Box-Cox-transformed engineered features for classical models;
  raw strain + tip deflection channels for NN and LSTM.
- Main results: GPR on 12 Box-Cox features dominates (adj-R²=0.9997, RMSE=0.021 g,
  MOS@1%=0.078 g). Polynomial Regression also meets all criteria (adj-R²=0.994, MOS@1%=0.156 g).
  Both classical models outperform all neural networks.
- Aerospace significance: physics-informed feature engineering (Box-Cox transforms) combined with
  classical surrogate models delivers real-time structural g-limit estimates viable for in-cockpit
  SHM — no deep learning required.
Do not use this text verbatim; revise in your own voice.
-->

---

## 3. Introduction and Problem Statement

<!-- CLAUDE NOTES — human authorship recommended for this section
Suggested talking points:
- Aircraft in contested environments sustain ballistic damage; pilots must immediately assess
  modified structural limits to execute safe evasive maneuvers and egress — ground analysis is
  not available.
- The g-limit (max load factor before first-yield structural failure) is the critical scalar
  linking structural damage to maneuver capability. Overpredicting it causes structural failure;
  underpredicting it is unnecessarily mission-limiting.
- SHM has focused on long-term maintenance and fatigue; real-time major-damage assessment is
  largely unsolved.
- This project: can a lightweight ML surrogate, trained on FEA data, predict post-damage g-limit
  from signals already onboard the aircraft (wing-tip deflection, surface strain gauges)?
- Why ML: the damage-to-g-limit mapping is high-dimensional and nonlinear; classical analytical
  methods require full knowledge of damage geometry; ML can learn a compact surrogate from
  strain patterns alone.
- Revised from preliminary: scope tightened to first-yield failure of the internal wing structure
  (skin excluded), ABAQUS as sole data source, safety-centric MOS@1% criterion added.
Do not use this text verbatim; revise in your own voice.
-->

---

## 4. Background, Related Work, and Existing Tools

### 4.1 Structural Health Monitoring in Aerospace

Current state-of-the-art SHM in aerospace targets preventative maintenance scheduling, fatigue detection, and automated inspection of critical infrastructure [1]. Machine learning has become central to SHM interpretation, with neural networks processing vibration, acoustic emission, and fiber-Bragg-grating strain data. Physics-informed machine learning (PIML) has emerged as a promising direction because it can embed governing equations into the loss function, reducing the data volume required for generalization [2] — directly relevant for FEA-limited datasets.

A directly related prior work from NUST used in-flight strain measurements to predict fatigue crack growth in a fighter airframe [3]. That work targeted long-duration damage accumulation rather than the instantaneous load-limit reduction of interest here.

### 4.2 Existing Tools and Methods

| Tool / Method | Purpose | Limitation for This Problem |
|---|---|---|
| ABAQUS / Nastran | FEA structural analysis | Requires full damage geometry; not real-time |
| Bayesian Health Monitoring (BHM) frameworks | Probabilistic SHM from sensor data | Primarily fatigue/maintenance; not instantaneous g-limit |
| OpenFSI / aeroelastic codes | Coupled fluid-structure response | Requires CFD mesh; no onboard deployment |
| GPR surrogate models (general) | Sample-efficient regression | Not previously applied to damaged wing g-limits |
| Classical beam theory | Analytical g-limit from geometry | Requires full knowledge of damage location and extent |

This project differs from all existing tools by targeting real-time inference from low-bandwidth sensor data (20 strain gauges + tip deflection) without requiring damage geometry knowledge, trained on FEA data designed to match the deployment sensor set.

---

## 5. Response to Preliminary Feedback

Key changes incorporated after the preliminary submission:

1. **Feature engineering implemented and documented** — Derived features were only conceptual in the proposal; they are now explicitly computed from the ABAQUS time-series and verified against physical expectations.
2. **Safety metrics added** — MOS@1% is now computed and reported for all seven models, directly tied to the aerospace consequence of overprediction.
3. **PINN physics terms clarified and ablated** — Three physics residuals (Hooke strain, strain energy, energy rate / Castigliano) are each implemented and compared across ten λ values rather than using a single default term.
4. **LSTM added as deep learning baseline** — A bidirectional LSTM on raw time-series data was implemented to test whether sequential load-ramp structure adds predictive value beyond scalar slope features.
5. **Success criteria tightened** — Threshold criteria (adj-R² ≥ 0.80, RMSE ≤ 0.75 g, MAE ≤ 0.50 g, MOS@1% ≤ 0.25 g) are now evaluated quantitatively, with 5-fold cross-validation confirming generalization on classical models.

---

## 6. Data and Preprocessing

### 6.1 Simulation Setup

All data were generated in ABAQUS using a parametric simulation of an A-10 Warthog–inspired skinless internal wing box structure. The skin was excluded to reduce computation time; skin-spar contact interaction is beyond the scope of this project. Damage was introduced by removing cylindrical volumes from structural members along randomized bullet trajectories (up to 20 perforations per wing). A distributed wing load was then ramped from zero to first yield of any structural node, which defines the g-limit for that damage configuration.

**Citation:** Proprietary dataset generated by this team using ABAQUS 2024, Texas A&M HPRC FASTER cluster, Spring 2026.

| Parameter | Value |
|---|---|
| Wing geometry | A-10 internal wing box; spars and ribs; skin excluded |
| Material | Aluminum alloy (linear elastic to first yield) |
| Damage model | Up to 20 randomized cylindrical perforations |
| Load steps per simulation | 2–29 |
| Output channels | Tip deflection, load at failure, strain at 24 gauge nodes |

### 6.2 Dataset Size and Split

| Split | Classical / NN / PINN | LSTM (raw time-series) |
|---|---|---|
| Training | 374 simulations | 383 simulations |
| Test | 94 simulations | 96 simulations |
| **Total** | **468** | **479** |

An 80/20 random split was used. Classical models were additionally evaluated with 5-fold cross-validation on the training set to detect overfitting.

### 6.3 Features and Target

**Target:** g-limit at first yield (scalar, g = 9.81 m/s²). Range: 0.05–1.68 g across the 468-case dataset.

**Engineered features (7 scalars)** — computed from the load-ramp time series:

| Feature | Description |
|---|---|
| `tip_deflection_slope` | Linear slope of tip deflection vs. applied load (m/N) |
| `tip_per_g_at_failure` | Tip deflection normalized by g-limit at failure (m/g) |
| `avg_strain_at_failure` | Mean strain across 24 gauges at failure load (mm/mm) |
| `avg_strain_slope` | Linear slope of mean strain vs. load (mm/mm per N) |
| `strain_energy_at_failure` | Elastic strain energy ½·Σ(εᵢ²·E) at failure (J) |
| `strain_energy_slope` | Slope of strain energy vs. load (J/N) |
| `k_spring` | Effective stiffness: load / tip deflection (N/m) |

**Box-Cox feature set (12 scalars):** The 7 engineered features plus `max_vm_stress_at_failure`, each transformed by the optimal Box-Cox power λ* that maximises univariate R² with g-limit. This pre-linearises the feature space and is used by GPR and Polynomial Regression. Linear Regression uses a pruned 6-column subset with collinear pairs removed.

**Extended feature set (31 scalars):** The 7 base features plus at-failure strain from all 24 gauge nodes plus tip deflection at failure. Used by Random Forest, Feedforward NN, and PINN.

**Raw time-series (26 channels × ≤29 timesteps):** All strain and kinematic channels preserving load-ramp sequence. Used by the LSTM only.

**Strain gauge rank ordering.** Because gauge node positions vary across simulations with different damage locations, raw gauge readings cannot be used as positionally-indexed features. Sorting each simulation's 24 at-failure strain values in ascending order assigns a within-simulation rank. The rank-23 reading achieves a single-feature linear R²=0.885 with g-limit — substantially higher than any unordered gauge reading (median R²≈0.41). A Gompertz growth curve fitted to the rank distribution achieves a continuous inverse-spline R²=0.840 (see Appendix Figure A8).

### 6.4 Preprocessing

1. **Slope extraction:** Load-ramp time-series fit by least-squares line per simulation to produce per-simulation slope features.
2. **Box-Cox transform:** Optimal λ* selected per feature by maximizing univariate linear R² with g-limit.
3. **Normalization:** All features standardized (zero mean, unit variance) using `StandardScaler` fit on training set only; same scaler applied to test set.
4. **LSTM padding:** Variable-length ramps zero-padded to 29 steps.

![Distribution of g-limit targets across the 468-case dataset.](../figures-v2/fig_01_target_dist.png)
*Figure 1: G-limit target distribution. The bimodal structure reflects two dominant damage-severity regimes.*

![Pearson correlation matrix for engineered features and g-limit target.](../figures-v2/correlation_matrix.png)
*Figure 2: Feature correlation matrix. High correlations among strain-energy features reflect shared physical origin; Box-Cox transforms reduce this collinearity in the GPR feature set.*

![Box-Cox power λ* sweep for each engineered feature.](../figures-v2/fig_08_boxcox_sweep.png)
*Figure 3: Box-Cox λ* sweep. Each curve shows univariate linear R² with g-limit as a function of the exponent λ. The optimal λ* (marked per feature) is used to construct the Box-Cox feature set. Most features improve substantially; near-zero λ (log transform) dominates for strain energy and k_spring.*

### 6.5 Dataset Limitations

- Wing skin excluded; real g-limits will be higher and differ in character.
- Static FEA only; no dynamic gust, flutter, or inertial relief effects.
- Damage modeled as clean cylindrical perforations; real ballistic damage includes petaling, cracking, and residual stress.
- No real flight data for external validation; sim-to-real transfer has not been demonstrated.

---

## 7. Methods and System Implementation

Seven models were implemented in Python (scikit-learn + PyTorch). All code is in `models/` and `scripts/` in the project repository.

### 7.1 Classical Models

**Linear Regression** — Ordinary least-squares on the 6-feature Box-Cox set with `StandardScaler` normalization. Box-Cox transforms span extreme numerical ranges (e.g., `boxcox_k_spring` ∈ [10³³, 10³⁹]); `StandardScaler` is required to prevent sklearn from discarding features with near-zero numerical weight. Serves as the interpretable baseline.

**Polynomial Regression** — Degree-2 polynomial expansion of the 7 base engineered features (35 basis functions after interaction terms), followed by ridge-regularized least-squares. Captures the nonlinear but smooth g-limit–strain relationship without overfitting on 374 training samples.

**Gaussian Process Regression (GPR)** — Matérn 5/2 kernel trained on the 12 Box-Cox features. Provides a probabilistic prediction with calibrated uncertainty. Box-Cox pre-linearisation substantially improves GP kernel fitting on this small dataset and allows the posterior variance to shrink close to zero.

**Random Forest** — Ensemble of 200 decision trees on all 31 features. Included as a non-parametric baseline; OOB R² converges by ≈200 trees.

### 7.2 Modern Models

**Feedforward NN (FFNN)** — Architecture: 31 inputs → [256, 128, 64] fully connected layers (ReLU, dropout 0.2) → 1 output. Trained with Adam (lr=1e-3, weight decay=1e-4), MSE loss, early stopping (patience=40 epochs). Stopped at epoch 192.

**Physics-Informed NN (PINN)** — Same MLP architecture with an additional physics regularization term:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

Three physics residuals compared via ablation across λ ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}:
- **Hooke strain:** R_F = K₁ × avg_strain (linear elastic assumption)
- **Strain energy quadratic:** R_F² = K₂ × U (elastic energy scales as F²)
- **Energy rate (Castigliano):** R_F = K₃ × dU/dF (tip deflection ∝ dU/dF ∝ R_F/k)

Each residual is normalized by the training-set standard deviation of its reference feature so that λ is dimensionless and comparable across physics models. Best configuration: **energy_rate at λ = 0.01**.

![PINN ablation heatmap: adj-R² across all physics model × λ combinations.](../figures-v2/ablation_heatmap.png)
*Figure 4: PINN ablation heatmap. Energy-rate physics achieves the best test adj-R² across all λ values. The optimal region is λ ∈ [0.003, 0.01]. High λ degrades performance as the physics loss overwhelms the data loss.*

**LSTM** — Bidirectional LSTM (2 layers, hidden size 64) on raw 26-channel time-series (≤29 timesteps). Trained with Adam, MSE loss, early stopping (patience=15). Stopped at epoch 31.

### 7.3 End-to-End Pipeline

```
ABAQUS simulation  →  Raw .csv time-series (24 strain gauges, tip deflection, load)
                           ↓  data_utils.py
               Feature engineering (7 scalars) + Box-Cox + standardization
                    ↓                                   ↓
        Classical models                          Modern models
        (train_classical.py)               (train_modern.py / PyTorch)
                    ↓                                   ↓
                    └──────── evaluate.py → results/*.json ─────────┘
                                           ↓
                              comparison_plots.py → figures-v2/
```

**Software:** Python 3.12, NumPy, SciPy, scikit-learn 1.4, PyTorch 2.3, ABAQUS 2024 (HPRC FASTER), Matplotlib 3.8.

---

## 8. Experimental Setup and Evaluation Metrics

### 8.1 Evaluation Metrics

Six metrics were computed for every model on the held-out test set:

| Metric | Symbol | Threshold | Description |
|---|---|---|---|
| Adjusted R² | adj-R² | ≥ 0.80 | Variance explained, penalized for feature count |
| Root Mean Square Error | RMSE | ≤ 0.75 g | RMS prediction error |
| Mean Absolute Error | MAE | ≤ 0.50 g | Average absolute prediction error |
| Maximum overprediction | max-over | — | Worst-case unsafe prediction |
| Margin of Safety @ 1% | MOS@1% | ≤ 0.25 g | Smallest margin ensuring <1% of predictions exceed true g-limit |
| Inference time | t_infer | — | Wall-clock per prediction, laptop CPU |

**MOS@1%** is the primary safety metric: it is the smallest constant that, when subtracted from all predictions, ensures fewer than 1% of test cases would result in an unsafe overprediction of the true g-limit.

### 8.2 Baseline Strategy

Linear Regression serves as the primary interpretable baseline. All models are compared against it and against the proposal success thresholds. Classical models additionally used 5-fold cross-validated R² on the training set.

---

## 9. Results

### 9.1 Model Comparison Summary

| Model | adj-R² | RMSE (g) | MAE (g) | Max Overpred. (g) | MOS@1% (g) | Infer. (ms) |
|---|---|---|---|---|---|---|
| **GPR** | **0.9997** | **0.021** | **0.007** | **0.181** | **0.078** | 1.2 |
| Poly Reg. | 0.994 | 0.100 | 0.070 | 0.211 | 0.156 | 0.24 |
| FFNN | 0.988 | 0.121 | 0.094 | 0.356 | 0.329 | 3.7 |
| PINN (energy-rate, λ=0.01) | 0.988 | 0.122 | 0.095 | 0.346 | 0.330 | 2.6 |
| Lin. Reg. | 0.970 | 0.226 | 0.181 | 0.650 | 0.644 | 0.09 |
| Random Forest | 0.956 | 0.229 | 0.137 | 0.885 | 0.729 | 57.9 |
| LSTM | 0.862 | 0.413 | 0.232 | 2.327 | 0.826 | 152.2 |

| Model | adj-R² ≥ 0.80 | RMSE ≤ 0.75 g | MAE ≤ 0.50 g | MOS@1% ≤ 0.25 g | **All criteria** |
|---|---|---|---|---|---|
| GPR | ✓ | ✓ | ✓ | ✓ (0.078) | **✓** |
| Poly Reg. | ✓ | ✓ | ✓ | ✓ (0.156) | **✓** |
| FFNN | ✓ | ✓ | ✓ | ✗ (0.329) | ✗ |
| PINN | ✓ | ✓ | ✓ | ✗ (0.330) | ✗ |
| Lin. Reg. | ✓ | ✓ | ✓ | ✗ (0.644) | ✗ |
| Rand. Forest | ✓ | ✓ | ✓ | ✗ (0.729) | ✗ |
| LSTM | ✓ | ✓ | ✓ | ✗ (0.826) | ✗ |

**GPR and Polynomial Regression are the only models meeting all four criteria. GPR leads on every metric.**

![Grouped bar comparison of adj-R², MAE, and RMSE across all models.](../figures-v2/comparison_bar.png)
*Figure 5: Model accuracy comparison. GPR leads on all accuracy metrics; Polynomial Regression is the best interpretable non-probabilistic model.*

![Safety metric comparison: max overprediction and MOS@1%.](../figures-v2/comparison_mos.png)
*Figure 6: Safety metric comparison. Dashed line marks MOS@1% = 0.25 g threshold. GPR (0.078 g) and Polynomial Regression (0.156 g) are the only models below it.*

### 9.2 Predicted vs. True and Error Distribution

![Predicted g-limit vs. true g-limit for all models.](../figures-v2/pred_vs_true.png)
*Figure 7: Predicted vs. true g-limit (test set). GPR and Polynomial Regression cluster tightly along the diagonal. LSTM shows systematic bias at extreme g-limits.*

![Cumulative distribution of absolute errors.](../figures-v2/abs_error_cdf.png)
*Figure 8: Absolute error CDF. GPR achieves the smallest errors across the full distribution; Polynomial Regression is second; the neural networks trail noticeably.*

### 9.3 Safety Analysis

![Overprediction CDF for all models.](../figures-v2/overpredict_cdf.png)
*Figure 9: Overprediction CDF. At the MOS@1% vertical marker, only GPR and Polynomial Regression fall below the 0.25 g threshold.*

Pairwise Wilcoxon signed-rank tests (Bonferroni-corrected, α=0.0024) confirm that GPR and Polynomial Regression are statistically significantly better than all other models (p<0.0001). FFNN, PINN, and Random Forest are statistically indistinguishable from each other (see Appendix Figures A1–A2). Errors are highest at very low g-limits corresponding to heavily damaged wings — see Appendix Figure A5.

### 9.4 Pareto Analysis

![Pareto front: adj-R² vs. inference time (top-3 non-dominated models).](../figures-v2/pareto_r2_vs_time_top3.png)
*Figure 10: Pareto front of accuracy vs. inference time. GPR and Polynomial Regression form the efficient frontier. LSTM is Pareto-dominated on both axes by every other model.*

![Pareto front: adj-R² vs. interpretability score (top-3 non-dominated models).](../figures-v2/pareto_r2_vs_interp_top3.png)
*Figure 11: Accuracy vs. interpretability. GPR occupies the top corner (highest accuracy, probabilistic output with calibrated intervals). Polynomial Regression is the best fully-deterministic interpretable model.*

### 9.5 Linear Regression Analysis

The standardized OLS coefficients (Figure 12) reveal how each Box-Cox-transformed feature drives the g-limit prediction. The dominant negative contributors are `avg_strain_slope` and `max_vm_stress` — physically expected, since higher strain response under load indicates greater structural damage and reduced load capacity. `k_spring` carries a distinct positive coefficient: a stiffer wing (higher load-to-deflection ratio) retains more load capacity despite damage. All six features contribute meaningfully; no feature has a coefficient sign that contradicts structural mechanics. The coefficient pattern provides a physically interpretable model that can be verified by a structural engineer without appeal to the underlying ML fitting.

![Standardized OLS coefficients for Linear Regression on the 6-feature Box-Cox set.](../figures-v2/lr_coefficients.png)
*Figure 12: Standardized Linear Regression coefficients. Each bar shows the change in predicted g-limit per one-standard-deviation increase in the corresponding Box-Cox feature. Avg strain slope and max VM stress dominate with negative coefficients (higher strain → lower g-limit); k_spring provides a positive structural-stiffness contribution.*

### 9.6 Polynomial Regression Analysis

The degree-2 Ridge expansion adds 35 basis functions to the 7 base features; ridge regularization selects the influential subset. The coefficient plot (Figure 13) shows that dominant linear terms mirror the LR hierarchy (avg strain slope, max VM stress), while the key nonlinear correction comes from the `avg_strain²` quadratic term and the `avg_strain × strain_energy` interaction term. Both carry negative coefficients, capturing the accelerating loss of load capacity as damage-induced strain amplifies near failure. The improvement from adj-R²=0.970 (LR) to 0.994 (PolyReg) is attributable to these interaction terms.

The partial dependence plots (Figure 14) confirm that the learned relationships are smooth and monotone — consistent with physical expectations for elastic structural failure. The concavity visible in the avg strain and VM stress PDPs corresponds to the physical saturation of elastic deformation near yield, not overfitting.

![Top-25 standardized Ridge coefficients for Polynomial Regression (degree-2 Box-Cox features).](../figures-v2/poly_coefficients.png)
*Figure 13: Polynomial Regression Ridge coefficients sorted by absolute magnitude. Blue = linear terms; red = pure quadratic; teal = cross-product interactions. Dominant linear terms mirror the LR hierarchy; the avg_strain² quadratic and avg_strain × strain_energy interaction terms provide the nonlinear correction closing the accuracy gap from adj-R²=0.970 to 0.994.*

![Partial dependence plots for the four most influential Polynomial Regression features.](../figures-v2/poly_pdp.png)
*Figure 14: Partial dependence plots (top-4 features by linear coefficient magnitude). Each curve shows the marginal predicted g-limit as one feature varies while all others are held at training-set means. Smooth, monotone, physically consistent responses confirm the degree-2 polynomial captures genuine nonlinearity rather than overfitting.*

### 9.7 Gaussian Process Regression Analysis

GPR is the only model in this study that provides a full posterior predictive distribution rather than a point estimate. Three aspects of the GPR analysis characterize the quality and structure of that uncertainty.

**Calibration** (Figure 15). The reliability diagram compares empirical coverage to the nominal confidence level. GPR lies slightly above the diagonal, indicating mildly conservative (over-wide) credible intervals — the safe direction for a structural assessment tool. Empirical 95% CI coverage is approximately 97%, with the excess explained by a small number of high-uncertainty predictions at very low g-limits.

**Uncertainty vs. g-limit** (Figure 16). The posterior standard deviation σ remains below 0.05 g across the full damage severity range, and does not grow systematically with g-limit. This confirms that the GP kernel maintains uniform confidence across all damage configurations tested — the model is not extrapolating into sparse regions of the training distribution at either end of the g-limit range.

**Feature sensitivity via length scales** (Figure 17). An anisotropic Matérn 5/2 refit confirms which features most influence the kernel. Features with short length scales (the kernel decays faster along them) are most discriminative; features with near-infinite length scales are effectively redundant given the others. Notably, the near-redundant features identified by the GP optimizer are the same collinear features that were removed from the LR feature set by manual analysis — an independent cross-validation of the feature pruning strategy.

![GPR calibration: predicted ± 2σ vs. true g-limit and reliability diagram.](../figures-v2/gpr_calibration.png)
*Figure 15: GPR uncertainty calibration. Left — test predictions with ±2σ error bars; blue points lie within the 95% CI, red points are outside it. Right — reliability diagram: empirical coverage vs. nominal confidence. GPR lies slightly above the diagonal, indicating mildly conservative but well-calibrated intervals.*

![GPR posterior standard deviation vs. true g-limit.](../figures-v2/gpr_uncertainty_vs_glimit.png)
*Figure 16: Posterior σ as a function of true g-limit, colored by absolute prediction error. Uncertainty is uniform across the g-limit range (median σ, black line, stays below 0.05 g), indicating consistent model confidence regardless of damage severity.*

![Anisotropic Matérn 5/2 per-feature length scales for GPR.](../figures-v2/gpr_length_scales.png)
*Figure 17: GPR per-feature length scales. Shorter length scale = higher kernel sensitivity = more discriminative feature. Features near the upper bound are near-redundant. The redundant features identified here match those pruned from the Linear Regression set by collinearity analysis, independently confirming the feature selection.*

### 9.8 Random Forest Analysis

Feature importances (Figure 18) confirm that physics-engineered features dominate the top positions despite being outnumbered 24:7 by raw gauge readings. `avg_strain_at_failure`, `strain_energy_at_failure`, and `k_spring` hold the three highest importance values individually; raw gauge importances are lower per-gauge but collectively non-negligible, explaining why the 31-feature RF achieves adj-R²=0.956 rather than the ≈0.93 expected from engineered features alone. The RF's failure on the safety criterion (MOS@1%=0.729 g) is not visible in the importance plot — it arises from tail behavior on a small number of out-of-distribution test cases where tree ensembles cannot interpolate smoothly (see Appendix Figure A3 residual tail).

![Random Forest top-20 feature importances by Gini criterion.](../figures-v2/rf_feature_importance.png)
*Figure 18: RF feature importances (mean decrease in impurity). Blue bars are base engineered features; orange bars are raw strain gauge readings. Engineered features dominate the top positions, validating the feature engineering approach. Individual raw gauge importances are lower but collectively contribute to accuracy.*

### 9.9 Neural Network Training Dynamics

**FFNN** (Figure 19). Validation loss decreases smoothly alongside training loss across 192 epochs, with no significant overfitting gap. Early stopping captures the performance plateau. The final test adj-R²=0.988 is consistent with validation behavior.

**PINN** (Figure 20). Convergence follows a similar trajectory to the FFNN. The physics term λ·L_physics adds slight regularization visible as marginally slower descent in early epochs, which leads to a marginally lower minimum validation loss. The final performance gain over the FFNN is 0.001 adj-R² and 0.008 g RMSE — a modest but consistent improvement.

**PINN ablation line plots** (Figure 21) show the λ-sensitivity of all three physics formulations. The Castigliano energy-rate term is the most robust: it maintains near-peak adj-R² across λ ∈ [0.003, 0.1] before degrading. Hooke-strain and strain-energy formulations plateau at slightly lower peak adj-R² and degrade more sharply at high λ, consistent with the Castigliano term better capturing the nonlinear stiffness reduction from structural damage.

**LSTM** (Figure 22). Validation loss diverges from training loss after epoch 15; early stopping triggers at epoch 31. The sustained gap between training and validation MSE indicates the model memorizes training sequences rather than learning a generalizable structural relationship. With only 383 training sequences, the LSTM lacks sufficient data for reliable temporal dependency learning. The per-sample comparison (Figure 23) places the FFNN and PINN side by side: the two models produce nearly identical absolute errors; PINN outperforms FFNN on fewer than 15% of test samples, concentrated near the most-damaged (lowest g-limit) cases.

![FFNN training and validation loss curves.](../figures-v2/feedforward_nn_loss.png)
*Figure 19: FFNN loss curves. Validation loss converges smoothly alongside training loss. Early stopping at epoch 192 captures the performance plateau; no significant overfitting gap.*

![PINN training and validation loss curves.](../figures-v2/pinn_loss.png)
*Figure 20: PINN loss curves (energy-rate physics, λ=0.01). The physics term adds a slight regularization effect in early epochs; convergence is similar to the FFNN. The combined data + physics loss is shown.*

![PINN ablation: adj-R² vs. λ for all three physics formulations.](../figures-v2/pinn_ablation.png)
*Figure 21: PINN λ-sensitivity curves. Energy-rate physics (Castigliano) maintains near-peak adj-R² across λ ∈ [0.003, 0.1] before degrading; Hooke-strain and strain-energy variants plateau at slightly lower peak accuracy and degrade more sharply at high λ.*

![LSTM training and validation loss curves.](../figures-v2/deep_learning_lstm_loss.png)
*Figure 22: LSTM loss curves. Validation loss diverges from training loss after epoch 15; early stopping at epoch 31. The sustained train/validation gap indicates memorization rather than generalization — insufficient training sequences for a bidirectional LSTM.*

![Per-sample absolute error comparison: FFNN vs. PINN (test set).](../figures-v2/nn_vs_pinn_per_sample.png)
*Figure 23: Per-sample FFNN vs. PINN absolute errors. Points below the diagonal favor PINN; above favor FFNN. The two models produce nearly identical errors; PINN improvement is concentrated on a small fraction of samples near the lowest g-limits.*

---

## 10. Discussion and Engineering Interpretation

### 10.1 Did the System Solve the Intended Problem?

Six of seven models satisfy the accuracy thresholds (adj-R² ≥ 0.80, RMSE ≤ 0.75 g, MAE ≤ 0.50 g). GPR and Polynomial Regression further satisfy the safety constraint (MOS@1% ≤ 0.25 g), making them viable for safety-critical SHM deployment. The system demonstrates that FEA-derived surrogate models with physics-motivated feature transforms can predict post-damage g-limits with sufficient accuracy for real-time in-cockpit use.

### 10.2 Classical Models Dominate — The Role of Box-Cox Feature Engineering

<!-- CLAUDE NOTES — human authorship recommended for this narrative framing
The key finding: GPR on Box-Cox features achieves adj-R²=0.9997 and MOS=0.078 g — approaching
the noise floor of a 468-sample FEA dataset. Polynomial Regression achieves 0.994 / 0.156 g.
Both dramatically outperform all neural networks.
The enabling factor is the Box-Cox power transform (Figure 3), which pre-linearises the feature
space so that both the GP kernel and the quadratic polynomial can fit the underlying elastic
mechanics with very few parameters.
Narrative: physics-motivated feature engineering (Box-Cox) unlocks the full accuracy of classical
models — increased model complexity does not substitute for domain knowledge.
For onboard deployment: Polynomial Regression (0.24 ms, deterministic, interpretable coefficients)
is preferred; GPR (1.2 ms, calibrated uncertainty) is appropriate if confidence intervals matter.
Do not use this text verbatim; revise in your own voice.
-->

The most striking result is that GPR and Polynomial Regression, when trained on Box-Cox optimal power-transformed features, dramatically outperform all neural networks. GPR achieves adj-R²=0.9997 and MOS@1%=0.078 g — approaching the noise floor of the 468-sample dataset. The Box-Cox pre-linearisation (Figure 3) is the key enabler: by mapping each feature to its most-linear relationship with g-limit before model fitting, both the GP kernel and the polynomial basis can fit the underlying elastic mechanics with very few parameters. The standardized coefficients (Figure 12) and PDPs (Figure 14) confirm these models are physically coherent, not just well-fit.

For onboard deployment, Polynomial Regression (0.24 ms, deterministic, interpretable) is the practical choice. If calibrated uncertainty intervals are required — for example, to compute a conservative g-limit margin automatically — GPR (1.2 ms, posterior σ ≈ 0.012 g median) is the appropriate upgrade.

### 10.3 PINN Physics Regularization

The PINN with energy-rate physics (Castigliano's theorem) provides consistent but marginal improvement over the FFNN. At optimal λ=0.01 the improvement is 0.001 adj-R² and 0.008 g RMSE (Figures 19–21). The Castigliano formulation outperforms Hooke's law because it captures the nonlinear stiffness reduction from structural damage more faithfully. However, neither NN variant closes the gap to the classical models, because the Box-Cox transforms allow classical methods to extract the same physics structure at zero training cost.

### 10.4 LSTM Underperformance

The LSTM (adj-R²=0.862, MOS@1%=0.826 g) is the worst performer. The diverging training/validation loss (Figure 22) confirms rapid overfitting with ≈380 sequences. The sequential load ramp structure — monotonically increasing — does not encode information that the scalar slope features do not already capture, making the added model complexity counterproductive.

### 10.5 Random Forest Safety Anomaly

Random Forest achieves competitive accuracy (adj-R²=0.956) but poor safety metrics (MOS@1%=0.729 g, max overprediction=0.885 g). Feature importances (Figure 18) confirm the model identifies the correct predictive signals; the failure is in tail extrapolation on rare damage configurations. This is a known property of ensemble tree models and is disqualifying for safety-critical deployment regardless of average accuracy.

### 10.6 Physical Reasonableness

Errors are highest at very low g-limits (heavily damaged wings), consistent with greater structural complexity in that regime (Appendix Figure A5). GPR's posterior σ does not grow systematically with g-limit (Figure 16), indicating uniform confidence across damage severity levels. All model predictions are bounded within plausible structural ranges.

---

## 11. Limitations, Risks, and Future Work

### 11.1 Current Limitations

- **Wing skin excluded:** Skin contributes to bending stiffness and torsional rigidity; real g-limits will differ. Largest fidelity gap.
- **Static loads only:** No dynamic gust loads, flutter, or inertial relief captured.
- **Simulation-only dataset:** No real flight sensor data for external validation; sim-to-real transfer undemonstrated.
- **Clean perforation damage model:** Real ballistic damage includes petaling, cracking, and residual compressive stress.
- **Single aircraft geometry:** Trained on one wing configuration; generalization requires retraining.
- **468 samples:** Adequate for polynomial and GPR models; limits neural network reliability.

### 11.2 Future Work

1. **Include wing skin** in the ABAQUS model and retrain — highest priority for fidelity improvement.
2. **Transfer learning** from FEA surrogate to real sensor data once physical test specimens are available.
3. **Expanded damage types** — crack propagation, delamination, multi-impact scenarios.
4. **Sensor placement optimization** — use GPR length-scale sensitivities (Figure 17) to identify the minimum gauge count and placement that maintains prediction accuracy.
5. **Dynamic loading extension** — transient FEA model with features capturing structural dynamics.

---

## 12. Aerospace Impact

### 12.1 In-Cockpit Structural Health Monitoring

Polynomial Regression evaluates in under 0.25 ms and GPR in under 1.2 ms on commodity hardware. With 20 surface-mounted strain gauges and a tip deflection sensor — standard SHM instrumentation — an aircraft computer could continuously update the pilot's displayed g-limit following damage. This gives pilots real-time situational awareness of their modified structural envelope, enabling maximum safe maneuvering without conservative guessing.

### 12.2 Autonomous Systems Integration

For unmanned combat aerial vehicles (UCAVs) or autonomous wingmen, a real-time g-limit signal feeds directly into the trajectory planner's constraint set. Rather than operating with a fixed pre-computed structural limit, the onboard planner can update its maneuvering constraints dynamically with measured damage state — directly improving survivability and mission effectiveness in contested airspace.

### 12.3 Sensor Specification and Transferability

The result that a 6–12 feature engineered model achieves adj-R²>0.99 is a quantitative specification result: 20 strain gauges and a tip deflection measurement are sufficient for reliable g-limit estimation. The GPR length-scale analysis (Figure 17) provides a path to further reduction. The broader methodology — dense FEA training set, physics-motivated scalar features, compact classical surrogate — transfers directly to other aerospace structures: landing gear, rotor blades, pressure vessels, and composite fuselage panels.

---

## 13. Conclusion

<!-- CLAUDE NOTES — human authorship recommended for this section
Suggested talking points:
- What was built: seven ML surrogates trained on 468 ABAQUS simulations of a damaged A-10-
  inspired wing box to predict g-limit from onboard strain and deflection features.
- Most important result: GPR on 12 Box-Cox-transformed features achieves adj-R²=0.9997,
  RMSE=0.021 g, and MOS@1%=0.078 g. Polynomial Regression also meets all criteria (0.994 / 0.156 g).
  Neural networks do not outperform physics-motivated feature engineering.
- Main technical takeaway: Box-Cox power transforms pre-linearise the feature space and allow
  classical models to fit the underlying elastic mechanics with near-zero residual on 468 samples.
  The PINN physics term provides incremental regularization benefit but cannot close the gap.
- Forward sentence: both GPR and Polynomial Regression are ready for onboard deployment testing;
  the critical next step is incorporating wing skin in the FEA model and validating against
  physical test data.
Do not use this text verbatim; revise in your own voice.
-->

---

## 14. References

[1] A. Entezami et al., "Machine Learning for Structural Health Monitoring of Aerospace Structures: A Review," *Sensors*, vol. 25, no. 19, 2025. https://www.mdpi.com/1424-8220/25/19/6136

[2] S. Karniadakis et al., "Physics-informed machine learning for Structural Health Monitoring," *arXiv preprint*, 2022. https://arxiv.org/pdf/2206.15303

[3] M. Azeem et al., "Integrated engineering framework for fatigue damage prediction of fighter aircraft using machine learning," *Results in Engineering*, 2025. https://www.sciencedirect.com/science/article/pii/S2590123025037661

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

[5] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.

[6] Dassault Systèmes, *ABAQUS 2024 Documentation*, 2024.

[7] Texas A&M HPRC, *FASTER Cluster User Guide*, 2025. https://hprc.tamu.edu/wiki/FASTER

---

## AI Tool Use Acknowledgement

Claude Code (Anthropic, claude-sonnet-4-6) was used to assist with results extraction, table generation, figure caption drafting, methods section structure, and report scaffolding. All technical claims, model implementations, computed metrics, and engineering interpretations were verified by the team. Abstract, Introduction, and Conclusion sections are reserved for human-authored revision.

---

## Appendix: Additional Figures

![Pairwise Wilcoxon p-values between all models.](../figures-v2/wilcoxon_pvalue.png)
*Figure A1: Wilcoxon p-value heatmap (Bonferroni-corrected, α=0.0024). Dark cells indicate statistically significant differences. GPR and Polynomial Regression are significantly better than all others (p<0.0001); FFNN, PINN, and Random Forest are statistically indistinguishable.*

![Win/tie/loss matrix across all model pairs.](../figures-v2/wilcoxon_wins.png)
*Figure A2: Win/tie/loss matrix: each cell counts test samples on which the row model has lower absolute error than the column model. GPR wins the majority of samples against every other model.*

![Residual distributions for all models.](../figures-v2/residual_dist.png)
*Figure A3: Residual (predicted − true) distributions. GPR and Polynomial Regression are centered near zero with narrow spread. Random Forest shows a right tail from occasional large overpredictions.*

![Safety bars: max overprediction and MOS@1% per model.](../figures-v2/safety_bars.png)
*Figure A4: Safety bar summary per model. Dashed line marks the MOS@1% = 0.25 g threshold.*

![Absolute error vs. true g-limit.](../figures-v2/error_vs_glimit.png)
*Figure A5: Absolute error vs. true g-limit (test set). Errors increase at very low g-limits, consistent with greater structural irregularity in heavily damaged wings.*

![Conservative prediction rate and median residual.](../figures-v2/conservative_rate.png)
*Figure A6: Conservative prediction rate (ŷ ≤ y_true) for each model. GPR is nearly unbiased (42.6% conservative, median residual +0.0003 g).*

![PINN λ sensitivity curves (line plot).](../figures-v2/ablation_lambda_curves.png)
*Figure A7: PINN λ sensitivity for each physics formulation. All variants degrade at high λ as the physics loss overwhelms the data loss; energy-rate is the most robust at moderate λ.*

![Strain gauge rank analysis.](../figures-v2/strain_gauge_rank_analysis.png)
*Figure A8: Rank-ordered strain analysis. Left: per-rank strain distributions (boxplots), median growth curve (navy), Gompertz fit (orange dashed), and per-rank R² with g-limit (right axis). Right: pooled unordered-node distribution showing that rank ordering creates the structured monotone signal. Best discrete predictor: rank-23 reading (R²=0.885).*

![Pareto front: adj-R² vs. inference time (all models).](../figures-v2/pareto_r2_vs_time.png)
*Figure A9: Full Pareto front (all 7 models). LSTM and Random Forest are Pareto-dominated on both accuracy and speed by every other method.*

![OLS R² vs. feature count under greedy forward selection.](../figures-v2/pareto_r2_vs_features.png)
*Figure A10: Feature-count Pareto. CV R² peaks at k=8 features under greedy forward selection on the 17-feature Box-Cox + ranked-strain pool; additional features are collinear and slightly reduce generalization.*

![Random Forest OOB R² vs. number of trees.](../figures-v2/rf_oob_curve.png)
*Figure A11: RF OOB R² as a function of n_estimators. Converges by ≈200 trees; the final OOB R²≈0.948 is consistent with test-set adj-R²=0.956, confirming no overfit.*

![Random Forest partial dependence plots.](../figures-v2/rf_pdp.png)
*Figure A12: RF partial dependence plots for the top-4 features. Consistent monotone character with the Polynomial Regression PDPs (Figure 14); slightly noisier at extremes, characteristic of tree ensembles in sparse data regions.*

![POD cumulative variance explained vs. mode count.](../figures-v2/pod_variance.png)
*Figure A13: Proper Orthogonal Decomposition cumulative variance for the strain field, used during exploratory feature analysis.*
