# TimesFM research comparison

This extension tests three separate uses of TimesFM 3.0 for next-session open,
high, low and close. Volume stays a historical input, not a prediction target.
Existing FinSTAK results and source data are preserved.

## Completed GPU outcome

The A100 study completed 48 fits over three seeds, 12 native validation
configurations, and 4,032 test examples. All 165 automated tests and independent
model, label, PCA, native-replay and future-input audits passed.

Validation retained the original no-decay FinSTAK configuration. For each new
family, its validation-selected variant had the following 2019 test error:

| Model | OHLC MAE normalized by observed close (%) |
| --- | ---: |
| Original FinSTAK, no decay | 0.7112 |
| Frozen TimesFM encoder, local-only, 64 sessions | 0.7153 |
| TimesFM forecast prior + local FinSTAK correction | 0.7419 |
| Native TimesFM + historical numerical covariates | 0.8034 |

The strongest new candidate's MAE reduction versus the original was
−0.00410 percentage points (95% date-block interval −0.00851 to −0.00142).
Thus the tested integration did not improve the current model. At a matched
64-session encoder context, adding the graph changed error from 0.7153% to
0.7161%; its paired reduction interval crosses zero. Event-deletion sensitivity
demonstrates that the trained graph routes use event evidence, without
establishing forecasting value or real-world causation.

The readable report and retained artifacts are in
[`results/timesfm_v1/results.html`](results/timesfm_v1/results.html).
The original configuration remains the recommendation. These reused 2019
results support a pilot-specific conclusion, not a general claim about TimesFM.

1. Native zero-shot forecasts, with price-only, historical numerical,
   direct-event, and direct-plus-peer-event covariates.
2. A frozen TimesFM forecast prior corrected by a trainable FinSTAK. The local,
   direct-event, peer-graph, and no-decay versions use the same training budget.
3. Frozen TimesFM transformer states replacing the GRU. The final context patch
   is averaged across the four price variates and reduced to 32 dimensions by
   training-only whitened PCA. Four matched event ablations are trained.

## Evaluation policy

- Train 2016–2017, select checkpoints and context lengths on 2018, evaluate on
  the same 4,032 stock-date examples from the original 2019 FinSTAK experiment.
- Context candidates are 32, 64, and 128 observed sessions. The original GRU
  uses 30; this difference is disclosed rather than presented as matched inputs.
- Three paired seeds, at most 25 epochs, five-epoch validation patience.
- Primary event comparisons are graph versus local for each trainable route.
  Direct versus graph and learned decay versus no decay are secondary.
- Native inputs are log OHLC divided by the observed forecast-day close. No
  future target or realized future covariate enters forecasting.
- Newer listings use the available contiguous historical prefix. The native API
  accepts variable lengths; low-level encoder padding is explicitly masked.
  Internal missing sessions still fail rather than being filled silently.
- Optional past-only numerical channels contain normalized historical volume
  and the existing 14 causal technical indicators. Event channels contain six
  category masses and four training-only PCA semantic moments, with a fixed
  five-session half-life. Peer channels diffuse only those event features on
  the existing same-date rolling-correlation graph.
- The hybrid can grow a wick even when native candle projection yields zero
  width. High/low losses do not propagate through the open/close bounds.
- Native candle geometry repairs and unmodified raw forecasts are both saved.
- Metrics are normalized MAE/RMSE, per-price errors, direction, and tolerance
  hits. MAE is never converted to `100 - MAE` accuracy.
- Paired calendar-block intervals keep all stocks from a date together. Positive
  MAE reduction favors the candidate. Intervals are exploratory and unadjusted
  for multiple comparisons.

The reused 2019 outcomes and unknown foundation-model pretraining overlap prevent
claims of independent temporal confirmation or contamination-free zero-shot
performance. Historical news arrival times retain the original end-of-source-day
assumption. Event confidence is not treated as a calibrated probability. The
dataset remains a 16-stock pilot.

## Reproduction

Official source: https://github.com/google-research/timesfm . The initial source
revision is `e51928e27119cb17bebc005be2696b75e0a9e688`; the run plan records both
source and model-checkpoint revisions, input hashes, and implementation hashes.
TimesFM 3.0 weights have a separate non-commercial, non-production license.
The code does not change the existing shared Python environment.

On this machine use `/var/tmp/mshanthappa-robotics/venv/bin/python` and set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/tmp/finstak-timesfm-cache
export PYTHONPATH=/tmp/stak-graph-python-deps:/tmp/finstak-timesfm-upstream/src:/home/mshanthappa/STAK-Graph
```

Run the primary study into a new output directory:

```bash
python -m stak_graph.experiments.timesfm_run \
  --source /var/tmp/mshanthappa-stak-graph/model_sweep_v2 \
  --expanded /var/tmp/mshanthappa-stak-graph/expanded_v3 \
  --output /path/to/new/primary-run \
  --upstream /tmp/finstak-timesfm-upstream \
  --cache /tmp/finstak-timesfm-cache \
  --reference /home/mshanthappa/STAK-Graph/results/finstak_proposal_v1
```

Use `scripts/verify_timesfm.py` with `--source`, `--expanded`, and `--output`
to reconstruct all hybrid checkpoints, labels and metrics and perturb future
inputs. Resuming the primary runner requires identical code/data/configuration.

Then run the encoder study:

```bash
python -m stak_graph.experiments.timesfm_encoder_run \
  --source /var/tmp/mshanthappa-stak-graph/model_sweep_v2 \
  --expanded /var/tmp/mshanthappa-stak-graph/expanded_v3 \
  --output /path/to/new/encoder-run \
  --foundation-run /path/to/new/primary-run \
  --cache /tmp/finstak-timesfm-cache
```

Use `scripts/verify_timesfm_encoder.py` with the same source, expanded, output
and cache arguments to reconstruct models, recompute PCA, and independently
re-extract a fixed sample of transformer states from raw prices.

The expanded study runs this encoder command three times with `--context 32`,
`--context 64`, and `--context 128`, using a separate new output directory each
time. This produces 36 encoder fits plus 12 forecast-prior hybrid fits.
`scripts/summarize_timesfm.py --primary ... --encoders ... ... ... --output ...`
selects a common encoder context from local-only validation MAE for the matched
graph-effect comparisons, and separately reports per-mode context tuning.
The final model recommendation is selected by validation MAE, including the
original no-decay model as a fallback; it is saved before test predictions are
read by the summarizer. Validation metrics for the hybrid and original fallback
are reconstructed by `verify_timesfm.py`.

Temporary source, optional test dependencies, and model weights can be removed
by system cleanup; retained plans identify what to download again. Final reports
are delivered separately under `results/` with complete predictions/checkpoints.
