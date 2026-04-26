# AERO 489 — Implementation Next Steps

**Deadline:** 4/24 preliminary scripts · 4/27 training complete · 5/1 submission  
**Owner (modern ML):** Ian Wilhite — Models 5–7 + shared infrastructure

---

## Build Order

Work top-to-bottom. Each step unblocks the next.

---

### 1. `evaluate.py` — finish two stubs
**Files:** `evaluate.py`  
**Why first:** every model depends on this; both functions are simple.

- [ ] `mos_01()` — return `float(np.percentile(y_pred - y_true, 99))`, floored at 0
- [ ] `inference_time_ms()` — warm-up call, then `timeit` loop over `n_reps`, return `float(np.median(times_ms))`

---

### 2. `data_utils.py` — finish `load_timeseries()`
**Files:** `data_utils.py`  
**Why second:** only blocks Model 7; scalar models can start immediately after step 1.

- [ ] Load `features/time_series.parquet` with polars
- [ ] Split unique `sim_id`s 80/20 with `train_test_split(sim_ids, random_state=42)`
- [ ] For each split: group by `sim_id`, build padded array `(n_sims, MAX_SEQ_LEN=29, 26)` and target vector `y`
- [ ] Return `X_train, X_test, y_train, y_test` plus `seq_lens_train, seq_lens_test` (actual step counts before padding)

> **Note:** `seq_lens` are needed by the LSTM's `pack_padded_sequence`. Decide whether to return them as a 5th/6th value or pack them as a sidecar array. Whatever you choose, update `DeepLearning.fit()` and `train_modern.py` to match.

---

### 3. Classical models — validate the full pipeline
**Files:** `models/linear_reg.py`, `models/poly_reg.py`, `train_classical.py`

Start with `LinearReg` (simplest) to confirm data → model → metrics → JSON works end-to-end before building the others.

- [ ] `LinearReg.fit()` — Pipeline: `StandardScaler → LinearRegression`; 10-fold CV; store `self.cv_r2_`
- [ ] `LinearReg.predict()` — `return self._pipeline.predict(X)`
- [ ] `train_classical.py: train_and_save()` — load data, fit, score, print table, write `results/{name}.json`
- [ ] Smoke-test: `python train_classical.py --models lr` — confirm JSON appears in `results/`
- [ ] `PolyReg.fit/predict()` — same pattern + `PolynomialFeatures(degree=2)` + `Ridge`

---

### 4. Classical models — GPR and Random Forest
**Files:** `models/gpr.py`, `models/random_forest.py`

- [ ] `GPR.fit()` — default kernel: `ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)`, `normalize_y=True`, `n_restarts=5`; 10-fold CV; store scaler + GPR separately (sklearn pipeline wrapping GPR loses `return_std`)
- [ ] `GPR.predict_with_std()` — exposes posterior std for use as data-driven MOS (mention in report)
- [ ] `RandomForest.fit()` — `GridSearchCV` over `PARAM_GRID` with 5-fold inner CV; refit on full train; store `feature_importances_`
- [ ] Run `python train_classical.py` — all four models; confirm four JSONs

---

### 5. `FeedforwardNN` — baseline modern model
**Files:** `models/feedforward_nn.py`

Validates the PyTorch training loop before adding physics constraints.

- [ ] `_MLP.__init__()` — `nn.Sequential` with `[Linear→ReLU→Dropout] × n_layers → Linear(hidden[-1], 1)`
- [ ] `_MLP.forward()` — `return self.net(x).squeeze(-1)`
- [ ] `FeedforwardNN.fit()`:
  - Fit `StandardScaler`; split off 10% val set
  - `TensorDataset` + `DataLoader`
  - `Adam(lr, weight_decay)` + `CosineAnnealingLR(T_max=epochs)`
  - Training loop: forward → `MSELoss` → backward → step; track val loss; early stopping after `patience` epochs; restore best weights
- [ ] `FeedforwardNN.predict()` — scale → eval mode → numpy
- [ ] `train_modern.py: train_scalar()` — fit, score, save loss curve to `figures/feedforward_nn_loss.png`, write JSON

---

### 6. `PINN` — add physics loss
**Files:** `models/pinn.py`

Inherits the same training structure from step 5; only the loss function changes.

- [ ] `_estimate_k_eff()` — OLS through origin: `K_eff = Σ(RF · ε) / Σ(ε²)` from training data (use raw, unscaled `avg_strain_at_failure`)
- [ ] Confirm `_AVG_STRAIN_IDX = 2` matches `ALL_SCALAR_COLS` ordering in `data_utils.py`
- [ ] `PINN.fit()` — same loop as `FeedforwardNN`, add physics term:
  ```
  RF_pred      = g_pred * AIRCRAFT_MASS_KG * G_ACCEL / 2
  physics_res  = RF_pred / K_eff − avg_strain_batch   # should → 0
  total_loss   = mse_loss + λ * mse(physics_res, 0)
  ```
  > `avg_strain_batch` must come from **unscaled** features — extract before the `StandardScaler` transform, or store the raw column separately.
- [ ] Ablation: try `lambda_physics ∈ {0.01, 0.1, 1.0}`; keep best on val loss

---

### 7. `DeepLearning` (LSTM) — sequence model
**Files:** `models/deep_learning.py`

- [ ] `_LSTMModel.__init__()` — bidirectional LSTM + `nn.Linear(hidden*2, 1)`
- [ ] `_LSTMModel.forward()` — `pack_padded_sequence` → LSTM → index last real hidden state by `seq_lens - 1` → head
- [ ] `DeepLearning.fit()`:
  - Compute per-channel mean/std from non-padded steps; store for inference
  - `_SeqDataset` + `DataLoader(batch_size=32, shuffle=True)`
  - `Adam` + `ReduceLROnPlateau(factor=0.5, patience=10)`
  - Same early stopping pattern as FeedforwardNN
- [ ] `DeepLearning.predict()` — normalise → eval mode → batch forward → numpy
- [ ] `train_modern.py: train_timeseries()` — call `load_timeseries()`, fit, score, save loss curve, write JSON

---

### 8. `compare.py` — final assembly
**Files:** `compare.py`

Run after all seven result JSONs exist.

- [ ] `load_results()` — glob `results/*.json`, sort by `metrics["adj_r2"]` descending
- [ ] `print_table()` — 7-row table, all 6 metrics, ✓/✗ against §6.2 thresholds
- [ ] `plot_bar_chart()` — grouped bars for R², MAE, RMSE; threshold lines
- [ ] `plot_mos_chart()` — max_overpredict and MOS@1% per model; 0.25g threshold line
- [ ] `plot_residuals()` — 7-panel scatter grid (y_pred vs y_true); annotate R²/RMSE

---

## Success Criteria (§6.2)

| Metric | Threshold |
|---|---|
| Adjusted R² | ≥ 0.80 |
| RMSE | ≤ 0.75 g |
| MAE | ≤ 0.50 g |
| MOS @ 1% | ≤ 0.25 g |
| Train ≈ Test | no significant overfit |

---

## Open Questions

- **Aircraft mass:** `AIRCRAFT_MASS_KG = 16,500` is a placeholder. Confirm with Barrett/Adam — affects `g_limit` scaling for all models and the PINN physics constant.
- **Strain gauge locations:** `Sim-params.md` lists this as outstanding. Needed for any spatial feature engineering (e.g. distance-weighted averaging), but not blocking current feature set.
- **Node volumes:** `strain_energy_at_failure` is currently a proxy (Pa, not J). If node volumes become available from the ABAQUS output, update `feature_engineering.py` and re-run.
- **LSTM seq_lens interface:** decide whether `load_timeseries()` returns 6 values or packs `seq_lens` into the array. Must be consistent between `data_utils.py`, `deep_learning.py`, and `train_modern.py`.
