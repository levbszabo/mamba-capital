## Mamba Multihorizon FX Forecasting Plan (Hourly)

### 1) Objective
- **Goal**: Build a global, multivariate, multihorizon forecasting model using a Mamba-style sequence architecture over long hourly histories (up to ~3 weeks of context) to predict future log-returns at three horizons: 1h, 6h, 24h.
- **Outputs**: For each time step, predict a 3-dimensional target vector: [log_return_1h, log_return_6h, log_return_24h].
- **Scope**: Multiple FX pairs from ~2018–2025. Single global model across symbols with symbol-aware encoding.

### 2) Data Overview
- **Source files**: `mamba-capital/data/forex_cleaned/processed_*.csv` and/or `mamba-capital/data/forex_all/*_1hour_*.csv`.
- **Core columns available**: `symbol, ts_utc, step_id, open, high, low, close, volume, is_gap, ny_hour, ny_dow, ret_log_1h, ret_log_6h, ret_log_24h, vol_real_24h` (some files may have empty ret columns at the beginning due to horizon definition).
- **Granularity**: Hourly, in UTC timestamp with New York time features provided (`ny_hour`, `ny_dow`).
- **Symbols**: Multiple currency pairs (e.g., EURUSD, AUDUSD, NZDUSD, USDJPY...).

### 3) Targets
- **Primary task**: Predict future log returns at +1h, +6h, +24h relative to the end of each input window.
- **Definition**: \( r_{k}(t) = \log(close_{t+k}) - \log(close_{t}) \) for k ∈ {1, 6, 24}.
- **Availability**: If present in files (e.g., `ret_log_*`), we will validate correctness; otherwise compute during preprocessing.

### 4) Time Splits and Backtesting Windows
- **Temporal splits (no leakage)**:
  - Train: 2018-01-01 → 2022-12-31
  - Validation: 2023-01-01 → 2024-06-30
  - Test: 2024-07-01 → 2025-12-31
  - Rationale: Contemporary validation and recent test to assess live performance.
- **Walk-forward evaluation**: Rolling windows with re-fit or fine-tune on expanding history; report average metrics across folds and final fixed test period.

### 5) Preprocessing
- **Integrity checks**: Ensure monotonic timestamps per symbol, fill missing hours to a complete hourly grid per symbol; mark gaps via `is_gap` and do not forward-fill target-dependent features.
- **Target computation** (if needed): Compute log returns using close-to-close prices ensuring lookahead-only from t to t+k.
- **Feature creation** (minimal v1):
  - Price-derived: log-price \(\log(close)\), hourly log-return, high-low range, close-open, realized volatility features (e.g., `vol_real_24h` if present, else compute rolling std of returns).
  - Volume transforms: log-volume and z-score per symbol.
  - Calendar: categorical `ny_hour` (0–23) and `ny_dow` (0–6), plus `is_gap`.
- **Normalization**:
  - Per-symbol robust scaling for continuous features (fit on train, apply to val/test).
  - Targets optionally standardized per-symbol (fit mean/std on train); store scalers for inverse-transform when reporting metrics.

### 6) Encodings ("one encoding" for hour, day-of-week, and symbol)
- **Categorical embeddings** (learned):
  - Symbol embedding: size E_sym (e.g., 32–64)
  - NY hour embedding: size E_hour (e.g., 8–16)
  - NY day-of-week embedding: size E_dow (e.g., 4–8)
- **Composition into a single context vector**: Sum or concatenate [sym, hour, dow] embeddings followed by a linear projection to `d_model` so that the model receives a single unified encoding per token. Use summation plus projection initially for parameter efficiency and stable training.

### 7) Sequence Construction
- **Input window length**: L = 512 tokens (~21.3 days) to cover ≥ 3 weeks while remaining efficient; later tune in {384, 512, 768}.
- **Sampling stride**: Train with stride S ∈ {1, 4, 8}; validation/test use stride 1 for dense evaluation.
- **Per-sample composition**:
  - X: continuous features over [t−L+1, …, t] and the unified categorical encoding for each time step.
  - y: vector of future returns at horizons {1, 6, 24} relative to t.
- **Global batching**: Sample sequences across all symbols to encourage cross-asset generalization.

### 8) Model Architecture (Mamba-style)
- **Backbone**: Mamba (selective state space model) stack; efficient for long context sequences.
- **Input embedding**: Linear projection from feature dimension to `d_model` plus addition of unified categorical context vector.
- **Positional treatment**: Mamba inherently models order; include learned positional bias if needed via a lightweight additive term.
- **Depth/width (v1)**: `d_model` ∈ {256, 384}, `n_layers` ∈ {6, 8}, dropout 0.1–0.2.
- **Output head** (multihorizon multitask):
  - v1: Independent Gaussian per horizon → predict mean and log-variance for each horizon (6 outputs total). Loss is summed NLL with horizon weights.
  - v2 (optional): Trivariate Gaussian with low-rank or Cholesky parameterization to capture cross-horizon correlation; adds covariance terms to the head.
- **Regularization**: Dropout, weight decay, layernorm; optional token-level stochastic depth for deeper stacks.

### 9) Loss, Objective, and Weights
- **Primary loss**: Negative log-likelihood (Gaussian) per horizon; sum with weights.
- **Horizon weights**: Start with w = [1.0, 0.8, 0.6] for [1h, 6h, 24h]; tune by validation. Alternative: scale by inverse horizon (1/k) or by target std to balance magnitudes.
- **Auxiliary losses** (optional):
  - Directional binary cross-entropy on sign(y) with small weight (e.g., 0.1) to stabilize directional skill.
  - Variance penalty or calibration regularizer if over/under-confident.

### 10) Training Regimen
- **Optimizer**: AdamW with weight decay 0.01–0.05.
- **LR schedule**: Cosine decay with warmup (e.g., 1k–5k steps), or one-cycle.
- **Batching**: Batch size tuned to GPU memory; gradient accumulation to emulate larger effective batch.
- **Precision**: Mixed precision (fp16/bf16) if hardware permits.
- **Stability**: Gradient clipping (e.g., 1.0), EMA of weights (optional).
- **Early stopping**: Monitor validation NLL and RMSE (1h) with patience.
- **Seeding**: Global seeds and deterministic dataloader ordering for reproducibility.

### 11) Evaluation & Metrics
- **Point-forecast metrics (per horizon)**:
  - RMSE, MAE, and standardized RMSE (after inverse-scaling if targets standardized).
  - R² and Pearson/Spearman correlation.
  - Directional accuracy: P[sign(ŷ) = sign(y)].
- **Probabilistic metrics**:
  - NLL per horizon; CRPS (optional); calibration curves/PIT histograms.
  - Coverage of 50/80/95% prediction intervals vs. nominal.
- **Trading-oriented diagnostics (sanity checks)**:
  - Simple sign-based strategy gross PnL and Sharpe per horizon with equal sizing, no transaction costs (for model intuition only).
  - Turnover and hit-rate by hour-of-day and symbol.
- **Reporting**: Metrics aggregated overall, by symbol, and by calendar buckets (hour, DOW). Include confidence intervals via block bootstrapping over weeks.

### 12) Backtesting Protocol
- **Static split evaluation**: Train on train; select on validation; report final on hold-out test period.
- **Walk-forward**: Rolling or expanding window retraining; evaluate sequentially on next block; aggregate statistics. Maintain a fixed hyperparameter set (from validation) for fairness.
- **Leakage control**: All scalers and derived features fitted on train windows only; no lookahead in feature construction.

### 13) Hyperparameter Search
- **Search space**:
  - Sequence length L ∈ {384, 512, 768}
  - d_model ∈ {256, 384, 512}
  - n_layers ∈ {4, 6, 8}
  - Dropout ∈ [0.0, 0.3]
  - Horizon weights and loss variants (independent vs. correlated head)
  - Optimizer LR ∈ [1e-5, 5e-4], weight decay ∈ [0.0, 0.05]
- **Strategy**: Bayesian search (e.g., 40–80 trials) constrained by compute; early stop unpromising runs using ASHA/median pruning.

### 14) Dataset Implementation Details
- **Data loader**: Symbol-wise datasets concatenated into a global index. Each sample maps to a contiguous window for one symbol.
- **Collation**: Pad-free since all windows are fixed length. Maintain symbol IDs for embedding lookup.
- **Stratification**: Ensure each mini-batch mixes symbols and calendar contexts.
- **Caching**: Precompute indices for windows; optional on-disk caches of normalized tensors to speed up training.

### 15) Inference & Serving
- **Batch inference**: Given latest L hours for each symbol, output mean and variance for each horizon.
- **Aggregation**: Store outputs with timestamps and symbols; optionally smooth means over short horizons for stability.
- **Monitoring**: Track live calibration drift and directional accuracy by recent window.

### 16) Artifacts & Reproducibility
- **Artifacts saved**: Model weights, optimizer state, config, symbol/hour/dow vocabularies, feature scalers, validation metrics, and commit hash.
- **Experiment tracking**: Log metrics, hyperparams, and artifacts; ensure results are reproducible from a single config file.
- **Environment**: Pin versions; document GPU/CPU requirements.

### 17) Risks & Mitigations
- **Non-stationarity/regime shifts**: Use rolling re-fits and recent validation; monitor degradation by year.
- **Long-horizon noise**: Emphasize probabilistic forecasts; avoid overfitting 24h horizon via loss weighting and regularization.
- **Calendar leakage**: Use provided `ny_hour`/`ny_dow` only; avoid future-aware holiday flags unless aligned to timestamp.
- **Gaps/market closures**: Model `is_gap`; ensure horizons are measured in trading hours, not absolute hours, to avoid bias when markets are closed. For v1, keep absolute hours aligned to provided dataset; document assumptions.

### 18) Deliverables & Directory Layout (proposed)
- **Files**:
  - `mamba-model/plan.md` (this document)
  - `mamba-model/config.yaml` or `.py` for all hyperparams and paths
  - `mamba-model/data_prep.py` for preprocessing and tensorization
  - `mamba-model/dataset.py` for windowed dataset and collate
  - `mamba-model/model.py` Mamba backbone and heads
  - `mamba-model/train_model.py` training loop, logging, checkpointing
  - `mamba-model/evaluate_model.py` static and walk-forward evaluation
  - `mamba-model/metrics.py` metric computations (point + probabilistic + trading diagnostics)
  - `mamba-model/utils.py` common helpers (scalers, seeds, IO)
  - `mamba-model/requirements.txt` pinned dependencies
- **Configs**: One canonical config with overrides for sequence length, symbol lists, and horizons.

### 19) Initial Hyperparameters (v1 defaults)
- **Data**: L=512, stride=4 (train), stride=1 (eval), batch_size as per GPU memory.
- **Model**: d_model=384, n_layers=6, dropout=0.1, independent Gaussian head.
- **Optimization**: AdamW (lr=2e-4, wd=0.01), cosine schedule with 2k-step warmup, max epochs=50, early stopping patience=5–8.
- **Loss weights**: [1.0, 0.8, 0.6].

### 20) Metrics to Report (table per horizon and overall)
- **Per horizon**: RMSE, MAE, R², Corr, Directional accuracy, NLL, 80/95% coverage.
- **Overall**: Macro-average across horizons; per-symbol breakdown.
- **Backtest**: Walk-forward averages with standard errors.

### 21) Roadmap
- **v1**: End-to-end pipeline with independent Gaussian heads, static split evaluation, and basic walk-forward.
- **v1.1**: Add calibration diagnostics and interval tuning.
- **v1.2**: Trivariate head with covariance; ablation on encodings (sum vs. concat+proj), and longer context.
- **v2**: Lightweight trading simulation with costs, position sizing from variance; model fine-tuning schedule.

### 22) Open Questions
- Should 6h/24h horizons be trading-hours adjusted across weekends/holidays or remain absolute hours per dataset? For now: follow dataset’s indexing.
- Include OHLC features beyond close-derived returns (e.g., HLC3, ATR) in v1 or later?
- Any additional instruments to include (commodities/indices) with a shared global model?


