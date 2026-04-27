# AERO 489 — Ian's ML TODO
> Role: Modern ML Approaches (Feedforward NN, PINN, Deep Learning/LSTM)
> Deadline: 5/1 submission | 4/29 check-in | TODAY 4/26

---

## Current results summary (what exists)

| Model         | adj-R²  | MAE (g) | RMSE (g) | max_overpredict (g) | MOS_1% (g) | infer (ms) | Pass? |
|---------------|---------|---------|----------|---------------------|------------|------------|-------|
| Feedforward NN | 0.988  | 0.094   | 0.121    | 0.356               | 0.329      | 3.7        | ✓*    |
| PINN          | 0.987   | 0.095   | 0.122    | 0.347               | 0.330      | 2.6        | ✓*    |
| LSTM          | 0.861   | 0.232   | 0.413    | 2.327               | 0.826      | 152        | ✗     |

*MOS_1% exceeds the 0.25g pass threshold — flag this in the report.
LSTM fails on MOS and max_overpredict; this is a key discussion point.

---

## IMMEDIATE — run before writing anything

- [ ] **Run PINN ablation** (`scripts/pinn_ablation.py`) — compares hooke_strain / energy_quad / energy_rate physics residuals
  - Output goes to `data-v2/pinn_ablation_output.txt` (currently empty)
  - Confirm before running: ~3× 25s = ~75s total — OK to proceed

---

## Figures needed (Ian's contribution to report)

- [ ] **Predicted vs True** scatter for each model (3 plots or 1 3-panel figure)
  - axes: true g-limit (x), predicted g-limit (y); draw y=x reference line
  - annotate R², RMSE on each panel
- [ ] **Error/residual distribution** histograms or violin plots (predicted − true) per model
  - highlight the MOS threshold at ±0.25g
- [ ] **PINN ablation bar chart** — adj-R² / RMSE / MOS for each physics residual variant
- [ ] **Training curves** — already exist in `figures-v2/` for all 3 models ✓
- [ ] **Pareto front #1**: adj-R² vs inference time (log scale) — all models (need classical team's numbers)
- [ ] **Pareto front #2**: adj-R² vs physical interpretability (ordinal axis: none / partial / full)
  - order: LSTM < Feedforward NN < PINN < GPR / physical model

---

## Report writing (Ian's sections)

### Methods section — what to write

- [ ] **Feedforward NN**: architecture (layers, activations, hidden dim), loss = MSE, optimizer, train/val split, early stopping if any
- [ ] **PINN**: same MLP + physics residual loss; explain all 3 physics models (Hooke's law, energy quadratic, Castigliano/energy rate); describe lambda weighting; note that physics_loss is normalised by training-set std
- [ ] **LSTM**: architecture, why sequence model was attempted (time-series data across 20 load steps), lookback window, input/output

### Results section — what to write

- [ ] Table of all 3 models with the 5 metrics from proposal: adj-R², MAE, RMSE, max_overpredict, MOS_1%, inference time
- [ ] Reference training curve figures and predicted vs true figures
- [ ] PINN ablation: which physics residual formulation wins and by how much

### Discussion section — key talking points (Ian drafts, human writes)

<!-- CLAUDE NOTES — human authorship recommended for this section
Suggested talking points:
- NN and PINN are nearly identical in performance — physics regularization didn't significantly change accuracy on this dataset; possible reasons: (1) data is sufficiently rich that the physics term is redundant, (2) lambda may need tuning, (3) the hooke_strain residual is a weak constraint compared to the full structural model
- LSTM dramatically underperforms (R²=0.861 vs 0.988) and has huge max_overpredict (2.33g). Likely reason: LSTM expects temporal autocorrelation between load steps, but the at-failure snapshots used as features don't carry step-order information that helps — LSTM is fighting the wrong problem structure
- MOS_1% for NN and PINN (~0.33g) exceeds the 0.25g goal. Discuss whether this is a data-size issue or a fundamental limit of the feature set
- Inference time: PINN (2.6ms) and NN (3.7ms) are real-time capable; LSTM (152ms) is borderline for embedded deployment
- Safety implication: for a safety-critical system, the model should never overpredict (dangerous). Max_overpredict matters more than R². The LSTM's 2.33g overpredict would be mission-critical.
-->

---

## Evaluation requirements (from proposal — make sure all are hit)

- [ ] Adjusted R² reported ← done in JSON, just needs to go in table
- [ ] RMSE reported ← done
- [ ] MAE reported ← done
- [ ] Maximum overprediction reported per model ← done
- [ ] Required MOS_1% reported per model ← done
- [ ] Inference time comparison (same hardware) ← done
- [ ] Train vs test performance comparison (to check for overfitting) ← need to extract train metrics, not just test

---

## Report structure checklist (from final report rubric — 100 pts)

| Section | pts | Ian's contribution |
|---------|-----|--------------------|
| Problem clarity | 15 | team writes intro; Ian reviews |
| Preliminary feedback response | 10 | team writes; note any scope changes |
| Background & related tools | 10 | team writes; Ian can cite PINN refs |
| **Technical approach** | 15 | **Ian writes NN/PINN/LSTM subsections** |
| Data & preprocessing | 10 | team writes; Ian notes feature set used |
| **Experimental setup & comparison** | 15 | **Ian writes eval metrics section + pareto fronts** |
| **Results & discussion** | 15 | **Ian writes modern ML results subsection** |
| Writing & professionalism | 10 | all |

---

## Lower-priority / stretch goals

- [ ] Strain gauge sensitivity analysis — what if you remove gauges? at what count does R² drop? (original TODO idea)
- [ ] 95th-percentile strain as single aggregate feature — quick test, interesting ablation
- [ ] Pareto front #3: R² vs development effort (ordinal) — may be too subjective

---

## Already done — do not re-implement

- [x] Feedforward NN model + training script
- [x] PINN model (3 physics residuals) + training script
- [x] LSTM model + training script
- [x] Loss curves for all 3 models (`figures-v2/`)
- [x] Feature engineering (`scripts/feature_engineering.py`)
- [x] EDA figures (dist, scatter, POD, correlation)
- [x] `compare.py` for cross-model evaluation
- [x] Result JSONs with y_pred/y_true arrays saved
