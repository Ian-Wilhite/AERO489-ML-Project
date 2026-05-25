# Machine Learning Prediction of Structural Load Limits in Damaged Wings

**Course:** AERO 489/689 — Introduction to Machine Learning for Aerospace Engineers, Spring 2026  
**Instructor:** Dr. Raktim Bhattacharya, Texas A&M University

---

## Team

| Team Member | Contribution |
|---|---|
| Barrett Brown | Scripting & Simulation — ABAQUS automation, data pipeline, batch execution on HPRC |
| Kirin Chadha | Classical ML — Linear Regression, Polynomial Regression implementation and evaluation |
| Malachi Drew | Aircraft Design & Modeling, Gaussian Process Regression, Random Forest |
| Ian Wilhite | Modern ML — Feedforward NN, PINN, LSTM, ablation study, comparison scripts |
| Adam Zheng | FEA Modeling — wing geometry, meshing, mesh refinement around damage sites |

---

## Abstract

Seven machine learning models were trained on 468 ABAQUS finite-element simulations of an A-10–inspired wing box sustaining randomized bullet-hole damage, to predict the post-damage g-limit (structural load factor at first yield) from onboard sensor signals alone — wing-tip deflection and strain readings from 20 surface-mounted gauges. Classical baselines (Linear Regression, Polynomial Regression, Gaussian Process Regression, Random Forest) and modern methods (Feedforward Neural Network, Physics-Informed Neural Network, LSTM) were evaluated across accuracy, safety, and inference-speed metrics.

The training set comprises 374 simulations (94 test); features are 12 Box-Cox power-transformed engineered scalars for GPR/Polynomial/Linear Regression, and 7 base engineered plus ranked strain channels for RF/FFNN/PINN. GPR on the 12-feature Box-Cox set is the top model overall (adj-R²=0.9997, RMSE=0.021 g, MOS@1%=0.078 g). Polynomial Regression also meets all safety criteria (adj-R²=0.994, RMSE=0.100 g, MOS@1%=0.156 g). Both substantially outperform all neural networks. The PINN with energy-rate physics (λ=0.01) is the best modern model (adj-R²=0.988, RMSE=0.122 g, MOS@1%=0.203 g). The LSTM underperforms without sufficient sequential data (adj-R²=0.862). The central finding is that physics-informed Box-Cox feature transforms allow classical surrogate models to achieve near-zero residual error on this dataset, with inference times below 1.2 ms — viable for real-time in-cockpit structural health monitoring without deep learning.

---

## Problem Statement

Aircraft operating in contested environments routinely sustain ballistic damage. Pilots must immediately know how their structural limits have changed to safely execute evasive maneuvers and plan their egress — pausing for ground analysis is not an option. Structural Health Monitoring (SHM) as a field has focused on long-term maintenance and fatigue; real-time in-flight damage assessment for major structural insult is largely unsolved.

The g-limit (maximum allowable load factor before first-yield failure) is the critical scalar quantity linking structural damage to maneuver capability. Overpredicting it can cause structural failure; underpredicting it is unnecessarily mission-limiting. Classical analytical approaches require full knowledge of the damage geometry — in combat, that geometry is unknown.

This project asks: can a lightweight ML surrogate, trained on FEA simulation data, predict the post-damage g-limit from signals already available on the aircraft (wing-tip deflection, onboard accelerometers, surface-mounted strain gauges) without any knowledge of the damage geometry? The mapping from distributed structural damage to g-limit is high-dimensional and nonlinear. We demonstrate that with physics-motivated feature engineering (Box-Cox power transforms on elastic-mechanical features), classical surrogate models can learn this mapping from 468 ABAQUS simulations with near-perfect accuracy and safety margins consistent with real-time cockpit deployment.

---

## Background

### Structural Health Monitoring in Aerospace

Structural Health Monitoring is an active research area in aerospace, with current state-of-the-art systems focused on preventative maintenance scheduling, fatigue detection, and automated inspection of critical infrastructure [1]. Machine learning has become increasingly central to SHM, with neural networks used to detect damage from vibration signatures, acoustic emission data, and fiber-Bragg-grating strain sensors.

Physics-informed machine learning (PIML) has emerged as a particularly promising direction for SHM because it enables data-efficient training by embedding governing equations into the loss function [2]. This is especially relevant here: an FEA-based dataset is inherently limited in size, and models that can leverage structural mechanics priors require fewer examples to generalize.

A directly related prior work from NUST used in-flight strain measurements to predict fatigue crack growth in a fighter airframe [3]. That work targeted long-duration damage accumulation rather than the instantaneous load-limit reduction needed here.

### Relevant Existing Tools and Methods

| Tool / Method | Purpose | Limitation for This Problem |
|---|---|---|
| ABAQUS / Nastran | FEA structural analysis | Requires full damage geometry; not real-time |
| BHM (Bayesian Health Monitoring) frameworks | Probabilistic SHM from sensor data | Primarily fatigue/maintenance; not instantaneous g-limit |
| OpenFSI / aeroelastic codes | Coupled fluid-structure response | Requires CFD mesh; no onboard deployment |
| GPR surrogate models (general) | Sample-efficient regression | Prior work not applied to damaged wing g-limits |
| Classical beam theory | Analytical g-limit from geometry | Requires full knowledge of damage location and size |

This project differs from all of the above by targeting real-time inference from low-bandwidth sensor data (20 strain gauges + tip deflection) without requiring knowledge of the damage geometry, and by training on FEA data that mirrors the expected deployment sensor set.

---

## Data & Preprocessing

### Data Source

All data were generated in ABAQUS using a parametric simulation of an A-10 Warthog–inspired skinless wing box structure. Damage was introduced by intersecting randomized bullet trajectories with the internal spar and rib geometry and removing the intersected material volume. The skin was omitted from the FEA model to reduce computation time and because skin-spar interaction requires higher-fidelity contact modeling beyond the scope of this project.

**Dataset:** Proprietary — generated by this team using ABAQUS 2024, Texas A&M HPRC (FASTER cluster), Spring 2026.

### Simulation Parameters

- **Wing geometry:** Based on A-10 wing box; internal spars and ribs modeled; skin excluded.
- **Material:** Aluminum alloy (linear elastic to first yield).
- **Damage model:** Up to 20 randomized bullet perforations per wing; each perforation removes a cylindrical volume from structural members.
- **Load application:** Distributed wing load ramped from zero to failure (defined as first-yield of any node).
- **Output per simulation:** Wing-tip deflection (≈20 load steps), strain at 20 gauge locations (≈20 steps), load at failure → g-limit.

### Dataset Size and Split

| Split | Classical / NN / PINN | LSTM (raw time-series) |
|---|---|---|
| Training | 374 simulations | 383 simulations |
| Test | 94 simulations | 96 simulations |
| **Total** | **468** | **479** |

An 80/20 random split was used. Classical models were additionally validated with 5-fold cross-validation on the training set.

### Features and Targets

**Target:** g-limit at first yield (scalar, units: g = 9.81 m/s²).

![Target g-limit distribution across the 468-simulation dataset.](figures/v2/fig_01_target_dist.png)
*Distribution of g-limit targets across the full dataset. The bimodal structure reflects two dominant damage-severity regimes.*

**Engineered features (7 scalars)** — computed from the load-ramp time series:

| Feature | Description |
|---|---|
| `tip_deflection_slope` | Linear slope of tip deflection vs. applied load (m/N) |
| `tip_per_g_at_failure` | Tip deflection normalized by g-limit at failure (m/g) |
| `avg_strain_at_failure` | Mean strain across 20 gauges at failure load (mm/mm) |
| `avg_strain_slope` | Linear slope of mean strain vs. load (mm/mm per N) |
| `strain_energy_at_failure` | Total elastic strain energy ½·Σ(εᵢ²·E·Vᵢ) at failure (J) |
| `strain_energy_slope` | Slope of strain energy vs. load (J/N) |
| `k_spring` | Effective wing stiffness: RF / tip_deflection (N/m) |

![Engineered feature scatter vs. g-limit.](figures/v2/fig_04_engineered_scatter.png)
*Scatter plots of each engineered feature against the g-limit target, illustrating nonlinear but monotone relationships.*

**Feature sets used per model:**

| Feature Set | # Features | Used By |
|---|---|---|
| `BOXCOX_COLS` (Box-Cox optimal power-transformed) | 12 | GPR, Polynomial Regression |
| `GREEDY_8_COLS` (greedy forward selection optimum) | 8 | Linear Regression |
| `BOXCOX_COLS + RANKED_STRAIN` | 17 | Random Forest, Feedforward NN |
| `BASE_ENGINEERED + RANKED_STRAIN` | 16 | PINN |
| Raw time-series (26 channels × ≤29 timesteps) | — | LSTM |

**Box-Cox transformation.** Most engineered features benefit from a power transform before regression. Box-Cox power λ\* was selected per feature by maximizing univariate R² with g_limit. The resulting `boxcox_*` columns substantially pre-linearize the feature space.

![Box-Cox optimal λ sweep: test R² vs. λ for each engineered feature.](figures/v2/fig_08_boxcox_sweep.png)
*Box-Cox power λ\* sweep for each engineered feature. Each curve shows how univariate R² changes as the exponent varies. The optimal λ\* is used to construct the `boxcox_*` feature columns.*

![R² heatmap: raw vs. ln / log₁₀ / exp / pow10 transforms for each engineered feature.](figures/v2/fig_07_transform_r2_heatmap.png)
*Heatmap of single-feature linear R² with g_limit across five transform types. Brighter cells indicate the best-performing transform per feature. Most features benefit from a log or Box-Cox compression; no single transform dominates across all features.*

**Strain gauge rank analysis.** Because the 24 strain gauge nodes have no fixed spatial correspondence across simulations (each simulation has a different damage location and loading configuration), raw gauge readings cannot be used as positionally-indexed features. To recover structure from this unordered set, we sort each simulation's 24 at-failure strain values in ascending order, assigning a within-simulation **percentile rank** (rank 1 = lowest strain, rank 24 = highest). This rank ordering is sufficient to produce a monotone, structured signal: the rank-ordered median growth curve rises smoothly across all 468 simulations, and the per-rank R² with g-limit is substantially higher than any unordered node-by-node R².

**Per-rank predictive power.** A single-feature linear regression on rank *k* strain achieves R² peaks near rank 4 (R²=0.798), dropping through the middle ranks (R²≈0.70), then rising to a global maximum at rank 23 (R²=0.885). **No combination of spline-transformed features exceeds the rank-23 single-gauge predictor.**

**Continuous rank feature via spline inversion.** Rather than using the discrete 24-point strain values directly, we fitted parametric and interpolating curves to the median rank-ordered growth profile and used each curve's inverse as a continuous feature. Eleven curve families were compared (logistic, cubic spline, PCHIP, Akima, least-squares B-spline, Gompertz, Richards, Weibull CDF, power law, exponential, Hill). The **Gompertz curve** (`y = a·exp(−b·exp(−c·x))`) best captures the shape (median inverse-feature R²=0.840 vs. 0.798 for logistic), with its asymmetric inflection matching the observed accelerating-then-saturating strain profile.

![Strain gauge rank analysis.](figures/v2/strain_gauge_rank_analysis.png)
*Left — per-rank strain distributions (blue boxplots), median growth curve (navy), cubic spline overlay (orange dashed). Right — pooled unordered-node strain distribution and per-node R² boxplot, demonstrating that rank ordering alone creates the structured monotone signal.*

### Preprocessing

1. **Slope extraction:** Time-series load-ramp data were fit with a least-squares line to extract per-simulation slope features.
2. **Strain energy:** Computed analytically from gauge readings, Young's modulus, and element volumes.
3. **Normalization:** All features were standardized (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler` fit on the training set only; the same scaler was applied to the test set.
4. **LSTM padding:** Variable-length load ramps were zero-padded to the maximum sequence length (29 steps) and packed for efficient batch processing.
5. **No data augmentation** was applied; the dataset represents a distinct ABAQUS simulation per sample.

### Dataset Limitations

- Skin contribution to structural integrity is excluded; real g-limits will differ.
- Only static FEA; no dynamic loading or fatigue effects.
- Damage is modeled as clean cylindrical perforations; real ballistic damage includes petaling, cracking, and residual stress.
- No real flight data available for external validation.

---

## Methods

Seven models were implemented in Python (scikit-learn + PyTorch) and evaluated on the same dataset. All code is in `models/` and `scripts/`.

### Classical Models (Baselines)

**Model 1 — Linear Regression** (`models/linear_reg.py`)  
Standard ordinary least-squares regression on physics-engineered features with `StandardScaler` normalization. Serves as the interpretable baseline.

> **Note on scaling requirement.** Box-Cox transforms span extreme numerical ranges (e.g., `boxcox_k_spring` ∈ [5×10³³, 10³⁹]; `boxcox_max_vm_stress` ∈ [10⁻¹⁰⁷, 10⁻¹⁰³]). Without `StandardScaler`, sklearn's OLS solver sets the corresponding coefficients to zero and silently discards those features. All results below use `StandardScaler` fit on the training set.

**Feature-set comparison (OLS R²):**

| Feature Set | # Features | In-sample R² | 5-fold CV R² |
|---|---|---|---|
| `ranked_strain_p23` (best single feature) | 1 | 0.885 | 0.884 |
| `RANKED_STRAIN` (3 rank percentiles + Gompertz shape) | 5 | 0.906 | 0.900 |
| `BOXCOX_COLS_LR` (6 non-collinear Box-Cox features) | 6 | 0.966 | 0.964 |
| Greedy-optimal (CV peak) | 8 | 0.978 | **0.976** |
| `BOXCOX_COLS` (all 12 Box-Cox features) | 12 | 0.976 | 0.972 |
| `BOXCOX_COLS + RANKED_STRAIN` (full pool) | 17 | 0.979 | 0.974 |

The CV R² peaks at k=8 under greedy forward selection. Beyond that, additional features are collinear residuals: in-sample R² increases by <0.001 while CV R² decreases slightly.

**Model 2 — Polynomial Regression** (`models/poly_reg.py`)  
Degree-2 polynomial expansion of the 7 engineered features (35 basis functions after interaction terms), followed by ridge-regularized least-squares. Captures the nonlinear but smooth g-limit–strain relationship.

**Model 3 — Gaussian Process Regression** (`models/gpr.py`)  
GPR with a Matérn 5/2 kernel trained on the 12 Box-Cox optimal power-transformed features. Provides a probabilistic prediction with calibrated uncertainty, naturally handling the small dataset. The Box-Cox feature set pre-linearises the input space, which substantially improves GP kernel fitting.

**Model 4 — Random Forest** (`models/random_forest.py`)  
Ensemble of 200 decision trees on all 31 features. Included as a non-parametric, high-variance baseline to assess the value of ensemble methods on this dataset size.

### Modern Models

**Model 5 — Feedforward Neural Network (FFNN)** (`models/feedforward_nn.py`)  
Architecture: 31 inputs → [256, 128, 64] fully connected layers (ReLU, dropout 0.2) → 1 output. Trained with Adam (lr=1e-3, weight decay=1e-4), MSE loss, early stopping (patience=40 epochs). Best validation loss at epoch 192.

**Model 6 — Physics-Informed Neural Network (PINN)** (`models/pinn.py`)  
Same MLP architecture as the FFNN with an additional physics regularization term added to the MSE loss:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

Three physics residuals were tested (ablation study below):
- **Hooke strain:** R_F = K₁ × avg_strain (linear elastic assumption)
- **Strain energy quadratic:** R_F² = K₂ × U (elastic energy scales as F²)
- **Energy rate (Castigliano):** R_F = K₃ × dU/dF (tip deflection ∝ dU/dF ∝ R_F/k)

The best-performing variant was **energy_rate** at **λ = 0.01** (adj-R²=0.988, RMSE=0.119 g). Each residual is normalized by the training-set standard deviation of the reference feature so that λ is dimensionless and comparable across physics models.

**Model 7 — Deep Learning LSTM** (`models/deep_learning.py`)  
Bidirectional LSTM (2 layers, hidden size 64) operating on raw time-series sensor data (26 channels, ≤29 timesteps). Intended to exploit sequential load-ramp structure. Trained with Adam, MSE loss, early stopping (patience=15 epochs). Stopped at epoch 31.

### End-to-End Pipeline

```
ABAQUS simulation
    ↓  (Barrett Brown / Adam Zheng)
Raw .csv time-series (strain gauges, tip deflection, load)
    ↓  data_utils.py
Feature engineering (7 scalars) + standardization
    ↓
Classical models (train_classical.py)   Modern models (train_modern.py)
    ↓                                        ↓
results/*.json   ←──── evaluate.py ─────────┘
    ↓
compare.py → figures/v2/, console table
```

### Software

- Python 3.12, NumPy, SciPy, scikit-learn 1.4, PyTorch 2.3
- ABAQUS 2024 (FEA, Texas A&M HPRC FASTER cluster)
- Matplotlib 3.8 (figures)

---

## Evaluation Metrics

Six metrics were computed for every model on the held-out test set:

| Metric | Symbol | Threshold | Description |
|---|---|---|---|
| Adjusted R² | adj-R² | ≥ 0.80 | Fraction of variance explained, penalized for feature count |
| Root Mean Square Error | RMSE | ≤ 0.75 g | RMS prediction error |
| Mean Absolute Error | MAE | ≤ 0.50 g | Average absolute prediction error |
| Maximum overprediction | max-over | — | Worst-case unsafe prediction |
| Margin of Safety @ 1% | MOS@1% | ≤ 0.25 g | Safety buffer such that <1% of predictions exceed true g-limit after subtraction |
| Inference time | t_infer | — | Wall-clock time per prediction on laptop CPU (comparative only) |

**MOS@1%** is the primary safety metric: it is the smallest constant margin that, when subtracted from all predictions, ensures fewer than 1% of test cases would result in an overprediction of the true g-limit. Smaller MOS means the model is both accurate and well-calibrated from a safety standpoint.

Classical models were additionally evaluated with 5-fold cross-validated R² on the training set to detect overfitting. A full PINN ablation ran a grid search over 3 physics models × 10 λ values (0.001 to 30).

---

## Results

### Model Comparison

| Model | Features | adj-R² | RMSE (g) | MAE (g) | Max Overpred. (g) | MOS@1% (g) | Infer. (ms) |
|---|---|---|---|---|---|---|---|
| **GPR** | **12 (BOXCOX)** | **0.9997** | **0.021** | **0.007** | **0.181** | **0.078** | **1.2** |
| Poly Reg. | 12 (BOXCOX) | 0.994 | 0.100 | 0.070 | 0.211 | 0.156 | 0.24 |
| Feedforward NN | 17 (BOXCOX+RANKED) | 0.9967 | 0.069 | 0.046 | — | 0.266 | 2.7 |
| PINN (energy-rate, λ=0.01) | 16 (BASE+RANKED) | 0.9964 | 0.073 | 0.053 | — | 0.203 | 3.3 |
| Random Forest | 17 (BOXCOX+RANKED) | 0.9909 | 0.116 | 0.065 | 0.532 | 0.348 | 54.5 |
| Linear Reg. | 8 (GREEDY-8) | 0.9803 | 0.180 | 0.133 | 0.535 | 0.442 | 0.23 |
| LSTM | 26 (raw time-series) | 0.862 | 0.413 | 0.232 | 2.327 | 0.826 | 152.2 |

**Proposal success criteria (✓ = meets, ✗ = does not meet):**

| Model | adj-R² ≥ 0.80 | RMSE ≤ 0.75g | MAE ≤ 0.50g | MOS@1% ≤ 0.25g |
|---|---|---|---|---|
| GPR | ✓ | ✓ | ✓ | ✓ (0.078) |
| Poly Reg. | ✓ | ✓ | ✓ | ✓ (0.156) |
| FFNN | ✓ | ✓ | ✓ | ✗ (0.266) |
| PINN | ✓ | ✓ | ✓ | ✓ (0.203) |
| Rand. Forest | ✓ | ✓ | ✓ | ✗ (0.348) |
| Lin. Reg. | ✓ | ✓ | ✓ | ✗ (0.442) |
| LSTM | ✓ | ✓ | ✓ | ✗ (0.826) |

GPR leads overall (adj-R²=0.9997, MOS=0.078 g). Three models meet all four criteria: GPR, Polynomial Regression, and PINN. FFNN misses the MOS threshold by 0.016 g. FFNN and PINN now substantially outperform Polynomial Regression on accuracy (adj-R²=0.9967/0.9964 vs. 0.9937). Random Forest improves dramatically after removing 74 redundant features (adj-R²: 0.956 → 0.991). GPR and Poly Reg remain on the accuracy–speed and accuracy–interpretability Pareto fronts; FFNN/PINN are dominated by GPR on both axes.

![Grouped bar comparison of adj-R², MAE, and RMSE across all seven models.](figures/v2/comparison_bar.png)
*Model performance comparison. GPR leads on all accuracy metrics; Polynomial Regression is the best interpretable non-probabilistic model.*

![Safety metric comparison: max overprediction and MOS@1% for all models.](figures/v2/comparison_mos.png)
*Safety metric comparison. The dashed line marks the MOS@1% = 0.25 g threshold from the proposal. GPR, Polynomial Regression, and PINN all fall below it.*

### Predicted vs. True and Residual Analysis

![Predicted g-limit vs. true g-limit for all models (test set).](figures/v2/pred_vs_true.png)
*Predicted vs. true g-limit plots for each model. GPR and Polynomial Regression show tight clustering along the diagonal. LSTM shows systematic bias at extreme g-limits.*

![Residual distributions for all models.](figures/v2/residual_dist.png)
*Residual (predicted − true) distributions. GPR and Polynomial Regression are centered near zero with narrow spread. Random Forest shows a right tail indicating occasional large overpredictions.*

![Cumulative distribution of absolute errors across models.](figures/v2/abs_error_cdf.png)
*CDF of absolute test errors. GPR achieves the smallest errors across the full distribution; Polynomial Regression is second best.*

### Safety Analysis

![Overprediction CDF: fraction of test predictions exceeding true g-limit by a given margin.](figures/v2/overpredict_cdf.png)
*Overprediction CDF. At the MOS@1% threshold (vertical marker), GPR (MOS=0.078 g) and Polynomial Regression (MOS=0.156 g) both satisfy the <1% overprediction requirement with a margin below 0.25 g.*

![Error as a function of true g-limit.](figures/v2/error_vs_glimit.png)
*Absolute error vs. true g-limit. Errors tend to be higher at very low g-limits (heavily damaged wings), consistent with greater structural complexity in that regime.*

![Conservative prediction rate and median residual per model.](figures/v2/conservative_rate.png)
*Conservative prediction rate (ŷ ≤ y_true) for each model. A rate of 50% indicates no systematic bias. GPR is nearly unbiased at 42.6% conservative with median residual +0.0003 g.*

### PINN Ablation Study

![PINN ablation heatmap of adj-R² across physics models and λ values.](figures/v2/ablation_heatmap.png)
*Heatmap of adj-R² for all physics model × λ combinations. The optimal region is λ ∈ [0.003, 0.01] for energy-rate physics.*

![PINN λ sensitivity curves.](figures/v2/ablation_lambda_curves.png)
*λ sensitivity curves. All physics models degrade at high λ (physics term dominates and overwhelms the data loss), confirming that the physics prior is a regularizer rather than a hard constraint.*

### Neural Network Training Curves

![Feedforward NN training and validation loss curves.](figures/v2/feedforward_nn_loss.png)
*FFNN loss curves. Validation loss converges smoothly; early stopping at epoch 192 prevents overfitting.*

![PINN training and validation loss curves.](figures/v2/pinn_loss.png)
*PINN loss curves. The physics term adds a small additional regularization effect; convergence behavior is similar to the FFNN.*

![Per-sample error comparison between FFNN and PINN.](figures/v2/nn_vs_pinn_per_sample.png)
*Per-sample absolute errors for FFNN vs. PINN. The two models perform nearly identically; PINN's physics penalty provides modest improvement on a small fraction of samples.*

### Pareto Analysis

![Pareto front: adj-R² vs. inference time.](figures/v2/pareto_r2_vs_time.png)
*Pareto front of accuracy vs. inference time. GPR and Polynomial Regression dominate the efficient frontier. The LSTM is Pareto-dominated on both axes.*

![Pareto front: adj-R² vs. model interpretability.](figures/v2/pareto_r2_vs_interp.png)
*Pareto front of accuracy vs. interpretability. GPR occupies the top-right corner (highest accuracy, probabilistic output). Polynomial Regression is the best fully-deterministic interpretable model.*

![Feature-count Pareto front: OLS R² vs. number of features.](figures/v2/pareto_r2_vs_features.png)
*OLS R² vs. feature count for greedy forward selection on the 17-feature pool. Circles = in-sample R²; squares = 5-fold CV R². The CV curve peaks at k=8 (R²=0.976), after which additional features are collinear and slightly reduce generalization.*

### Statistical Significance

Wilcoxon signed-rank tests were applied to all 21 pairwise model combinations on the same test set (n=94 matched predictions; LSTM uses n=96 via Mann-Whitney U due to different set size). Bonferroni correction was applied (α=0.0024 per pair).

![Pairwise Wilcoxon p-values between all models.](figures/v2/wilcoxon_pvalue.png)
*Wilcoxon p-value heatmap (|error| pairs). Dark cells indicate statistically significant differences. GPR and Polynomial Regression are significantly better than all other models (p<0.0001). Random Forest, FFNN, and PINN are statistically indistinguishable from each other.*

### GPR Uncertainty Quantification

A key advantage of GPR over all other models is that it provides a posterior predictive distribution, not just a point estimate.

![GPR predicted ± 2σ vs. true g-limit and calibration reliability diagram.](figures/v2/gpr_calibration.png)
*Left — test-set predictions with ±2σ error bars. Right — calibration reliability diagram: empirical coverage vs. nominal confidence level. GPR lies slightly above the diagonal, indicating mildly conservative (over-wide) intervals — the safe direction for structural assessment.*

![GPR posterior standard deviation vs. true g-limit.](figures/v2/gpr_uncertainty_vs_glimit.png)
*Posterior standard deviation σ as a function of true g-limit. The median σ stays below 0.05 g across the full range — consistent with the near-perfect point-prediction accuracy (RMSE=0.021 g).*

![GPR anisotropic Matérn kernel per-feature length scales.](figures/v2/gpr_length_scales.png)
*Per-feature length scales from an anisotropic Matérn 5/2 refit. Features with length scales near the upper bound are redundant given the other features in the set — an independent confirmation of the manual collinearity analysis.*

---

## Discussion

### Did the System Solve the Problem?

All seven models meet the threshold criteria for adj-R², RMSE, and MAE. Three models fully satisfy the safety constraint (MOS@1% ≤ 0.25 g): GPR (0.078 g), Polynomial Regression (0.156 g), and PINN (0.203 g). FFNN narrowly misses at MOS@1%=0.266 g. The system demonstrates that FEA-derived surrogate models with physics-motivated feature transforms can predict post-damage g-limits with sufficient accuracy and safety for real-time SHM applications, **provided the correct feature set is selected**. The updated feature sets (Box-Cox + ranked strain) close much of the accuracy gap between modern and classical models — FFNN and PINN now match or exceed Polynomial Regression in adj-R².

### The Dominance of Classical Models with Box-Cox Features

GPR, trained on 12 Box-Cox optimal power-transformed features, remains the unambiguous accuracy leader (adj-R²=0.9997, MOS@1%=0.078 g). However, the updated feature sets substantially close the classical–modern performance gap: FFNN and PINN on compact Box-Cox + ranked strain sets (17 and 16 features respectively) now outperform Polynomial Regression on adj-R² (0.9967/0.9964 vs. 0.9937). The key enabler is the same in both cases — pre-linearised features that reduce the learning problem to a near-linear regression. For the classical models, Box-Cox transforms do this analytically; for the neural networks, using 17 well-engineered features rather than 100 redundant transform variants gave a 6× better sample-to-feature ratio and a cleaner training signal.

For onboard deployment, Polynomial Regression (0.24 ms inference) remains the practical choice — deterministic, faster than the NNs by an order of magnitude, and with interpretable coefficients. PINN (3.3 ms, MOS=0.203 g) is now a viable alternative if a physics-regularized solution is desired. If uncertainty quantification is required, GPR is the appropriate upgrade (1.2 ms, calibrated confidence intervals).

### PINN Physics Regularization

The PINN with energy-rate physics (Castigliano's theorem) modestly improves over the FFNN baseline at low λ. At high λ the physics term overwhelms the data loss and performance degrades sharply. This suggests the physics prior is best used as a regularizer rather than a hard constraint given the limited dataset size. The Castigliano energy-rate formulation outperforms Hooke's law because it captures the nonlinear stiffness change due to structural damage more faithfully.

### LSTM Underperformance

The LSTM achieves adj-R²=0.862, RMSE=0.413 g, and MOS@1%=0.826 g — the worst of all models. This is attributable to three factors: (1) the raw time-series representation includes noise not present in the engineered features, (2) with ~380 training sequences the LSTM lacks sufficient data to learn temporal dependencies reliably, and (3) early stopping at epoch 31 indicates rapid overfitting. The sequential structure of the load ramp (load increases monotonically) does not appear to add predictive value that the scalar slope features do not already capture.

### Physical Reasonableness

Predictions are physically reasonable: g-limit predictions are bounded within plausible structural ranges (no negative g-limits except for a single linear regression outlier), and errors are highest at very low g-limits, corresponding to heavily damaged wings where the structural response is most irregular — consistent with physical intuition.

### Complexity–Accuracy–Safety Tradeoffs

The Pareto analysis confirms that increased model complexity does not improve the safety–accuracy tradeoff for this dataset. GPR and Polynomial Regression dominate the efficient frontier across all tradeoff axes. If uncertainty quantification is required, GPR is the appropriate choice; for deterministic onboard deployment, Polynomial Regression is preferred.

---

## Limitations & Future Work

### Current Limitations

- **No skin in FEA model:** The wing skin contributes to bending stiffness and torsional rigidity. Real g-limits will differ from skinless predictions. This is the largest fidelity gap.
- **Static loads only:** FEA models quasi-static load application; dynamic gust loads, flutter, and inertial relief are not captured.
- **Simulation-only dataset:** No real aircraft strain data exist for external validation. Sim-to-real transfer has not been demonstrated.
- **Clean perforation damage model:** Ballistic impact creates petaling, cracking, and residual compressive stresses that are not modeled.
- **Single aircraft geometry:** The model is trained on one wing configuration. Generalization to other aircraft requires retraining.
- **468 samples:** While adequate for polynomial and GPR models, this dataset size limits the depth and reliability of neural networks.

### Future Work

1. **Include wing skin** in the ABAQUS model and retrain. This is the highest-priority improvement.
2. **Transfer learning** from the FEA surrogate to real sensor data once physical test data become available.
3. **Expanded damage types** including crack propagation, delamination, and multiple simultaneous perforations.
4. **Sensor placement optimization** — use the trained GPR or PINN sensitivity to determine the minimum number and location of strain gauges needed to maintain prediction accuracy.
5. **Dynamic loads** — extend the FEA model to transient loading and retrain with time-series features that capture structural dynamics.
6. **VM stress trajectory feature** — extract `max_vm_stress_slope` from the time-series parquet. This scalar achieves R²=0.54 with g-limit individually and is expected to improve the greedy-optimal feature set beyond the current k=8 CV plateau.
7. **Spanwise damage centroid** — compute the load-weighted spanwise y-coordinate of the dominant strain gauges at failure: `Σ(εᵢ × yᵢ) / Σεᵢ`. This encodes whether failure originates inboard or outboard — structural mode-shape information not captured by any current scalar feature.
8. **Physics-motivated interaction terms** — include `ranked_strain_p23 × boxcox_strain_energy` and `ranked_strain_p24 × boxcox_k_spring` as cross features.

---

## Repository Structure

```
AERO489-Proj/
├── models/                  # Model implementations (sklearn + PyTorch)
│   ├── base.py
│   ├── linear_reg.py
│   ├── poly_reg.py
│   ├── gpr.py
│   ├── random_forest.py
│   ├── feedforward_nn.py
│   ├── pinn.py
│   └── deep_learning.py     # LSTM
│
├── scripts/                 # Training, evaluation, and plotting scripts
│   ├── data_utils.py        # Feature engineering and data loading
│   ├── train_classical.py
│   ├── train_modern.py
│   ├── evaluate.py
│   ├── comparison_plots.py
│   ├── pinn_ablation.py
│   ├── significance_tests.py
│   └── ...
│
├── results/                 # JSON metric files per model + PINN ablation CSV
│
├── features/
│   ├── v1/                  # Dataset round 1 (parquet) — initial pipeline test
│   ├── v2a/                 # Dataset round 2 — HPRC Machine A (parquet)
│   ├── v2b/                 # Dataset round 2 — HPRC Machine B (parquet)
│   └── v2/                  # Dataset round 2 merged/final (parquet) ← used for all results
│
├── figures/
│   ├── v1/                  # EDA figures for dataset v1
│   ├── v2a/                 # EDA figures for dataset v2a
│   ├── v2b/                 # EDA figures for dataset v2b
│   └── v2/                  # Final figures (used in report and this README)
│
├── data/
│   ├── v1/                  # Training run logs — dataset v1
│   ├── v2a/                 # Training run logs — dataset v2a
│   ├── v2b/                 # Training run logs — dataset v2b
│   └── v2/                  # Training run logs — dataset v2 final
│
├── docs/                    # Project proposal and supplementary docs
│   ├── AERO-489_Proeject_Proposal.pdf
│   ├── feature_walkthrough.md
│   └── Sim-params.md
│
└── reports/
    └── final_report_draft.md   # Full technical report
```

**Dataset versioning:** v1 was the initial pipeline test (the structural model's g-limit scale was off); v2a and v2b were round-two simulation batches split across two HPRC machines in HRBB; v2 is the merged final dataset used for all reported results.

---

## References

[1] A. Entezami et al., "Machine Learning for Structural Health Monitoring of Aerospace Structures: A Review," *Sensors*, vol. 25, no. 19, 2025. https://www.mdpi.com/1424-8220/25/19/6136

[2] S. Karniadakis et al., "Physics-informed machine learning for Structural Health Monitoring," *arXiv preprint*, 2022. https://arxiv.org/pdf/2206.15303

[3] M. Azeem et al., "Integrated engineering framework for fatigue damage prediction of fighter aircraft using machine learning," *Results in Engineering*, 2025. https://www.sciencedirect.com/science/article/pii/S2590123025037661

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

[5] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.

[6] Dassault Systèmes, *ABAQUS 2024 Documentation*. Vélizy-Villacoublay, France, 2024.

[7] Texas A&M HPRC, *FASTER Cluster User Guide*, 2025. https://hprc.tamu.edu/wiki/FASTER

---

## AI Tool Use Acknowledgement

Claude Code (Anthropic, claude-sonnet-4-6) was used to assist with: results extraction and table generation, figure caption drafting, methods section structure, and report scaffolding. All technical claims, model implementations, computed metrics, and engineering interpretations were verified by the team. The Abstract, Introduction, and Conclusion sections are marked for human-authored revision in the full report draft.
