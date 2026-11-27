# STAK-Graph

**Stacked Temporal And Knowledge Graphs** includes sourced event extraction, frozen event embeddings, causal stock-day graph construction, and reproducible real-data training and baseline evaluation. This is a research implementation; forecasting improvement must be assessed from the saved held-out results, not from the synthetic development examples.

**Current objective: predict the next trading day's open, high, low and close.**
The TimesFM 3.0 extension adds native covariate baselines, a frozen-forecast
FinSTAK correction, and a frozen-transformer encoder replacing the GRU.
The evaluation policy and reproduction commands are in
[`TIMESFM_RESEARCH.md`](TIMESFM_RESEARCH.md). Its measured results are reported
separately; using a foundation model alone is not evidence of improvement.
The completed [48-fit TimesFM comparison](results/timesfm_v1/results.html)
retained original no-decay FinSTAK: normalized OHLC MAE 0.7112%, versus 0.7153%
for the strongest new validation-selected encoder candidate. No improvement
was found in this pilot.

The November 2026 proposal adaptation is implemented in
`models/finstak.py` and `experiments/finstak_run.py`. It encodes bounded
stock-local histories before typed event attention, uses a zero-valued null
attention sink with learned category/relation decay, and then diffuses event
shocks across same-session correlations. It adds shared-price trailing
90-session OHLC normalization (expanding prefixes during warmup), keeps volume
as a historical input, and retains the valid four-price decoder. The attachment
inventory and implementation decisions are in
[`results/finstak_proposal_v1/implementation.html`](results/finstak_proposal_v1/implementation.html).
Run `python -m stak_graph.experiments.finstak_run --source
/var/tmp/mshanthappa-stak-graph/model_sweep_v2 --expanded
/var/tmp/mshanthappa-stak-graph/expanded_v3 --output /path/to/new/run`.
The default comparison uses three seeds, up to 20 epochs, event/no-event/no-decay
ablations, LSTM, and a project GCN–GRU comparator. Numerical tolerance hit rates
are reported alongside regression errors; they are never computed as 100 minus
MAE. This remains a pilot on the existing 16-stock panel, not the proposed
five-year S&P 500 evaluation. Existing frozen results are preserved.

The implementation is `experiments/next_day_sweep.py`; it uses a one-session
horizon and four continuous price outputs. SPY is absent from this task's
inputs, labels and metrics. Numerical forecasts are evaluated by per-feature
MAE/RMSE, with absolute next-close direction as a secondary metric. The earlier
five-session SPY-relative classification results below answer a different
research question and do not validate this objective.

The corrected comparison and example forecasts are in
[`results/next_day_ohlc_v1/results.html`](results/next_day_ohlc_v1/results.html).
Run `python -m stak_graph.experiments.next_day_sweep --source
/var/tmp/mshanthappa-stak-graph/model_sweep_v2 --expanded
/var/tmp/mshanthappa-stak-graph/expanded_v3 --output /path/to/fresh/run`, then
`python scripts/verify_next_day.py` with those same three arguments. Use the
isolated ML runtime and dependency settings documented below. Training uses
four-price regression; direction is computed afterward from predicted versus
observed own-stock close. Both last-OHLC and last-close persistence are reported.

The extraction component runs independently of embeddings and stock prices. Its local inference default is `Qwen/Qwen3-4B`; a Gemma instruction model or another model can also be served through the HTTP backend. The embedding stage uses `google/embeddinggemma-2` in text-only mode. These model choices have not been established as the best ones for financial forecasting.

## What is implemented

- Six event categories: earnings, guidance changes, acquisitions, financing, litigation, and operational disruptions.
- Separate extraction of explicitly stated supplier, customer, ownership, and competitor relationships.
- Schema-constrained generation on the local Transformers backend using XGrammar; schema requests on compatible HTTP servers such as vLLM.
- Character-preserving, overlapping chunks for long documents, with source-global quotation offsets.
- Mandatory timezone-aware publication and availability timestamps, normalized to UTC.
- Exact, dated company-alias resolution; ambiguous and unknown companies are flagged for review.
- Checks for invented quotations, unquoted company names, unsupported numerical magnitudes, unsupported event dates, and duration units.
- Checks for clear purchase-from/supply-to relationship directions; other phrasing is flagged for review.
- Raw-response audit trails, bounded format/evidence correction, response caching, exact output deduplication, quarantine, and run manifests.
- Separate fixture replay for CPU tests. Replay outputs explicitly carry `simulated: true`.

## Input

Supply one JSON object per line. Each document requires:

```json
{
  "document_id": "unique-source-document-id",
  "text": "The original article or filing section.",
  "source_url": "https://example.com/original-document",
  "published_at": "2026-10-01T20:30:00Z",
  "known_at": "2026-10-01T20:32:00Z"
}
```

Optional fields are `title`, `source_type`, and `is_synthetic`. Keep publication and first-availability timestamps separate; `known_at` must not precede publication. The pipeline does not fetch the URL or guess missing timestamps. Preserve the original text so evidence offsets remain meaningful.

The company CSV uses `company_id,name,aliases,valid_from,valid_to`. Separate aliases with `|`; validity dates use `YYYY-MM-DD` and are inclusive. Supply historical aliases rather than relying on current ticker strings. The example registry contains fictional `DEMO_` identities, not real securities.

## Installation and CPU demonstration

Inside a suitable Python 3.10+ environment:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m stak_graph.extraction schema
python -m stak_graph.extraction extract \
  --input examples/documents.jsonl \
  --registry examples/companies.csv \
  --backend replay \
  --replay-responses examples/replay_responses.json \
  --output results/my_replay_run
```

The three supplied documents and saved responses are synthetic. Replay tests application behavior without making model calls; it is not an extraction benchmark.

## Run a local model on the GPU

Install local inference dependencies in your own environment if necessary:

```bash
python -m pip install -e '.[local]'
python -m stak_graph.extraction extract \
  --input examples/documents.jsonl \
  --registry examples/companies.csv \
  --backend transformers \
  --model Qwen/Qwen3-4B \
  --model-cache /tmp/stak-graph-models \
  --cache /tmp/stak-graph-extraction.sqlite3 \
  --output results/my_gpu_run
```

The model is downloaded on first use. Add `--revision COMMIT` to pin weights and `--local-files-only` once downloaded. The actual local model revision, library versions, prompt/schema hashes, decoding settings, and source hash are recorded. Generation uses the non-thinking chat template and greedy decoding with an enforced JSON grammar. Prompts exceeding the input/context limit cause an explicit error rather than silent truncation.

On the implementation machine, the existing `/var/tmp/mshanthappa-robotics/venv/bin/python` has the required libraries and can replace `python` in these commands without installing into or changing that environment. GPU access is required for `--device cuda`. Model weights are kept in `/tmp` because the home filesystem is nearly full; that temporary cache may need to be downloaded again after cleanup or restart.

The schema mask uses ordinary Torch operations with XGrammar's matcher, so it does not require a Triton JIT build or system Python development headers.

## Use a model served locally

For an already-running compatible server, including vLLM serving an instruction model:

```bash
python -m stak_graph.extraction extract \
  --input examples/documents.jsonl \
  --registry examples/companies.csv \
  --backend http \
  --base-url http://127.0.0.1:8000/v1 \
  --model MODEL_NAME_ON_SERVER \
  --output results/my_server_run
```

This backend sends a `json_schema` response format and disables thinking through the server's chat-template settings. The server must support those settings and the chosen model. Optional endpoint authentication is read from `STAK_API_KEY`; credentials are not stored in provenance. A supplied HTTP model revision is operator-reported because the client cannot verify the server's weights.

## Outputs

Each run creates a **new** directory. An existing output directory is never overwritten.

| File | Contents |
| --- | --- |
| `documents.jsonl` | Source metadata, original text, and content hash |
| `events.jsonl` | Extracted events, participants, financial fields, and quotation offsets |
| `relationships.jsonl` | Explicit company relationships and their source evidence |
| `event_company_links.jsonl` | Resolved, directly stated participant associations that passed deterministic checks |
| `quarantine.jsonl` | Invalid source records, exhausted chunk failures, and rejected extractions |
| `audit.jsonl` | Raw model responses, errors, retry attempts, token usage, and cache hits |
| `manifest.json` | Model/prompt/schema provenance, processed-input checksum, registry checksum, configuration, and counts |

An event has one of three validation statuses:

- `evidence_checked`: schema, exact quotations, numerical surface checks, and identity resolution passed.
- `needs_review`: quotations passed but an identity, currency, unit, claim status, relationship phrasing, or another supported check needs review. These events are not exported as participant links.
- `rejected`: a deterministic evidence check failed. The record appears in quarantine, not the accepted event stream.

These statuses are not a semantic entailment guarantee. For example, an exact quotation can still fail to justify an event category or a company's role. `semantic_support_verified` remains `false`; human review and a labeled evaluation set are still needed.

An extracted occurrence date requires its own exact `occurrence_date_evidence` quotation. Explicit ISO dates and common full English dates are supported. A missing date is `null`, never filled from publication metadata. If format or deterministic evidence checks fail, the model gets at most one correction attempt by default; remaining unsupported records are quarantined. A run with invalid input or exhausted chunk failures exits with code 2. The manifest distinguishes completed, completed-with-errors, and aborted runs.

`confidence`, `expected_direction`, `half_life_days`, and `valid_until` remain `null`. A model's self-reported confidence is not calibrated, and a ten-day shutdown does not imply a ten-day stock-market half-life. Downstream experiments can use a clearly labeled fixed-weight baseline until a separately validated confidence model exists.

Participant links identify an event's stated participants; they do not establish stock-price causality. A customer relationship is stored separately and does not automatically turn the customer into an affected event participant. The graph stage below optionally projects one-hop business exposure as explicitly inferred connections.

The extractor preserves `known_at`. The implemented graph stage maps availability to an explicit trading-session cutoff and disallows unrestricted reverse paths. Exact duplicate outputs are collapsed within a document; semantic merging of syndicated stories across documents is not performed automatically.

## Reproducible extension points

- Prompt: `stak_graph/extraction/prompts/extract_v1.txt`.
- Shared schema and versions: `stak_graph/extraction/schema.py`.
- Local, HTTP, and replay model backends: `stak_graph/extraction/backends.py`.
- Evidence and numerical checks: `stak_graph/extraction/validation.py`.
- Batch orchestration and caching: `stak_graph/extraction/pipeline.py`.
- CLI and export contract: `stak_graph/extraction/cli.py`.

A first financial-data evaluation should use manually reviewed event and no-event documents, measure event/participant precision and recall, numerical correctness, and evidence support, and keep calibration examples separate from held-out evaluation. The supplied synthetic examples validate plumbing, not financial accuracy or forecasting improvement.

## Verified extraction run

The completed GPU development run is saved in `results/gpu_final/`:

| Check | Observed result |
| --- | --- |
| GPU | NVIDIA A100 80 GB |
| Local model | `Qwen/Qwen3-4B` |
| Model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Schema / prompt versions | `1.1.0` / `1.1.1` |
| Inputs | Three synthetic development articles |
| Events | Two: operational disruption and guidance change |
| Company relationship | Boreal Motors `customer_of` Acme Components |
| Participant associations | Two; the customer was not automatically added as an event participant |
| No-event article | Returned no events |
| Final quarantined records / failed chunks | Zero / zero |
| Extraction time after model setup | Approximately 38 seconds |
| Automated tests at the extraction milestone | 56 passing in the local inference environment |

The validation feedback corrected a reversed relationship during the shutdown example; the raw initial response and correction remain in `audit.jsonl`. Source hashes and quotation offsets were independently checked against the saved original text. The guidance values remained USD 5.0 billion and USD 4.5 billion, and an absent duration remained null.

These examples were used during development. They establish that the component runs and its checks work on these cases; they do not establish held-out financial extraction accuracy, calibrated confidence, or forecasting improvement. The complete machine-readable record is [the run manifest](results/gpu_final/manifest.json), and the extracted events are in [events.jsonl](results/gpu_final/events.jsonl).

## Event embeddings

The frozen text encoder receives the validated factual event description plus deduplicated supporting quotations, including evidence for amounts and duration. It does not receive future prices or return labels. An unchanged event is embedded once; daily persistence is handled by graph-edge weights, not repeated encoding.

The native Transformers implementation loads only EmbeddingGemma 2's text component, pools its projected token states using the attention mask, and normalizes the resulting vector. It uses the documented classification-task prefix and float32 weights. No Sentence Transformers installation is required by this backend.

In a suitable environment, install `.[embeddings]` if needed, then run:

```bash
python -m stak_graph.embeddings \
  --events results/gpu_final/events.jsonl \
  --output results/my_event_vectors \
  --model google/embeddinggemma-2 \
  --revision f9972ac22e696a105c482202646979e33e4b5a1e \
  --model-cache /tmp/stak-graph-models \
  --cache /tmp/stak-graph-vectors.sqlite3 \
  --device cuda
```

Use `--local-files-only` after downloading. The existing machine runtime described above also runs this command without being modified. The cache key includes the exact text, text-template version, model revision, pooling, prefix, dimension, and runtime versions.

Outputs are `vectors.f32` (little-endian float32), `index.jsonl`, `skipped.jsonl`, and `manifest.json`. Event IDs and text/source hashes link vectors back to their evidence. The vector loader checks dimensions, binary length, unique row IDs, and unit norms; the model rejects stale event/vector joins. Exact financial values, units, and periods remain in the sourced event records and graph metadata.

`--backend test-hash` is a CPU integration-test placeholder with `simulated: true`. It is not a semantic encoder and must not be presented as a financial embedding model.

## Causal daily graph construction

The market input uses two explicit CSV tables:

- Calendar: `session_date,open_at,close_at,prediction_at,is_synthetic`, with timezone-aware timestamps. A missing prediction cutoff defaults to the supplied close time.
- Prices: `company_id,session_date,open,high,low,close,volume,available_at,is_synthetic`.

The calendar is supplied rather than inferred from weekdays, so holidays, early closes, and daylight-saving changes can be represented explicitly. Daily bars must be complete by the declared prediction cutoff. Duplicate/revised bars and inconsistent OHLC values fail validation. Supply consistently adjusted data and a dated security registry for research runs.

```bash
python -m stak_graph.graph \
  --events results/gpu_final/events.jsonl \
  --relationships results/gpu_final/relationships.jsonl \
  --prices examples/prices.csv \
  --calendar examples/calendar.csv \
  --registry examples/companies.csv \
  --cutoff 2026-10-08T20:00:00Z \
  --output results/my_graph
```

The example prices and calendar are explicitly synthetic. Replace both tables for a real-data experiment.

Each StockDay contains 14 causal features: return, opening gap, candle body/range/wicks, log-volume change, trailing volatility, moving-average distance, and data-availability/gap indicators. Features use only the observed prefix. Missing sessions are not silently treated as consecutive days. Raw price levels are not used as cross-stock neural features.

The snapshot contains Company, Document, Event, and StockDay nodes. Temporal edges point forward. Event participation and business relations retain their sources. `ACTIVE_ON` edges are generated only for sessions when both the event and any relationship used in its path were known. News after a prediction cutoff first becomes eligible at the next supplied session. Events older than calendar coverage are excluded with a reason; their age is never reset to zero.

The initial event policy is a **fixed-weight baseline**: a five-session half-life, twenty-session TTL, direct weight 1.0, and optional one-hop supplier/customer propagation factor 0.5. These are configurable experimental priors, not calibrated probabilities or measured market-impact durations. `confidence` stays null unless a supplied value is explicitly marked calibrated. Expected stock direction remains unknown.

Use `--direct-only` for the no-propagation ablation. Other controls are `--lookback`, `--half-life-sessions`, `--ttl-sessions`, and `--propagation-weight`. Inferred links retain the business-relation ID and are distinguishable from direct participation. An event's physical duration never sets its market half-life automatically.

Company and Document paths are not automatically reversed into StockDay nodes. The graph auditor checks node/edge types, source availability, temporal direction, relationship-path timing, and finite features/weights. A future price, event, or later-discovered relation cannot be used to alter an earlier daily input.

Outputs are `snapshot.json` and an input/snapshot-checksum manifest. Expiration stops new event injection; a temporal model can still retain information from earlier days after an explicit edge expires.

## Graph and temporal model baseline

`EventGraphGRU` implements learned, relation-specific event-to-StockDay message passing followed by a forward-only GRU shared across stock timelines. It has separate message transforms for direct participation and inferred business exposure. Absolute confidence/decay weights remain effective because aggregation does not divide by their sum. This is a small graph baseline, not an HGT or FinGAT reproduction.

The present neural baseline consumes candle features, event text vectors, and event categories. Structured financial fields are retained in graph metadata for a subsequent explicit numeric-feature encoder. The output heads provide return scores, direction logits, and nonnegative volatility outputs. They require training against real targets before they can be interpreted as forecasts.

Run an architecture check with `.[model]` installed if necessary:

```bash
python -m stak_graph.models \
  --snapshot results/my_graph/snapshot.json \
  --embeddings results/my_event_vectors \
  --output results/my_model_check \
  --device cuda
```

This command runs a finite forward/backward check and one optimizer step on a diagnostic objective. It does not fit future-return labels, save a usable forecasting model, or report accuracy. Its output explicitly states `is_trained_forecaster: false` and `is_forecasting_evaluation: false`.

## Verified embedding/graph milestone

The embedding/graph milestone passed **100 automated tests**, including future-input invariance, after-cutoff/weekend handling, expiry, delayed-relation propagation, stale-vector rejection, nonzero graph gradients, and sensitivity to absolute edge weights. Setting all event weights to zero matches the graph-free event-input baseline numerically.

- `results/embeddings_gemma2/`: two real 768-dimensional EmbeddingGemma 2 vectors produced on the A100 from the synthetic development events. The encoder is real; the example source documents remain synthetic.
- `results/graph_final/`: 21 StockDay nodes across three fictional companies, two Event nodes, 59 total edges, ten direct active edges, and five inferred active edges.
- `results/graph_direct_only/`: the same market inputs with zero inferred active edges, for comparison.
- `results/model_gpu_smoke/model_smoke.json`: successful A100 forward/backward check, 15 active event edges, and nonzero gradients through the graph and temporal layers. The 84,035-parameter baseline is not a trained financial predictor.

Source remains small; downloaded weights stay in temporary storage. The synthetic checks above establish software behavior, not forecasting gains. The real-data experiment below is a separate retrospective pilot with explicit availability assumptions.

## Earlier GRU/TFT real-data forecasting experiment

The earlier GRU/TFT study passed **138 automated tests**, including same-day timing and TFT causality checks. The configuration is `configs/real_pilot_v3.json`. It compares compact TFT and GRU models under the same news timing. The previous two-day-delay study is archived; its three removed baselines are excluded from new training and evaluation.

The data source is [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), pinned to revision `bf9189c41527198897d1af3e17b1a0095279fc45`. Both complete upstream artifacts are checked against their published SHA256 values. Large downloads stay in `/tmp`; the final report directory preserves the small processed inputs needed to reproduce training without parsing the original news release again.

- Fixed universe: AAPL, MSFT, INTC, CSCO, JNJ, PFE, MRK, AMGN, JPM, BAC, WFC, GS, KO, PEP, WMT, MCD; SPY is the market benchmark.
- Prices: 2015-12 warmup through early 2020 label maturity; exact XNYS sessions, holidays, daylight-saving changes and early closes from `exchange_calendars`.
- News: 699 distinct real headlines. One hash-selected general headline plus one hash-selected event-keyword headline per company/month where available; sampling never reads future returns. This release has qualifying news for 12 of the 16 stocks.
- Availability: every sampled source timestamp is a date-only midnight placeholder. Headlines are assumed available at **23:59:59 UTC on the recorded date**, and forecasts use that end-of-day cutoff. Reported nonmidnight times are preserved when present. Bars remain available 15 minutes after the actual market close. Weekend/holiday news enters the next supplied trading decision. Source-date accuracy and historical arrival times are unverified.
- Target: stock log return minus SPY log return, from the following session's open to the fifth following session's close. Auxiliary targets are the sign of that excess return and log future five-session RMS volatility.
- Splits: train 2016–2017, validate 2018, test 2019. Removing labels that cross split boundaries leaves 7,856 training, 3,936 validation and 4,032 test examples, with 160 purged examples. The last test labels mature in early 2020.
- Inputs: 30 sessions of 14 candle features plus four causal SPY features. Every neural condition receives company identity. TFT's known-future input is only the relative horizon index, with no future observed prices or news.
- Selection: three fixed seeds, fixed optimizer/loss settings, early stopping on validation multitask loss, and no test-based parameter search. Test metrics are generated after all checkpoints are selected.

There are five retained graph scenarios, each run with **GRU and TFT**, for ten conditions and thirty selected checkpoints across three seeds: price plus rolling correlation, first-day event injection, temporal event graph, event plus correlation graph, and a historical-content negative control. The correlation graph takes up to three positive neighbors from the previous 60 completed returns. The negative control substitutes already-available event content while preserving timing and connections. Training-mean, linear-price, and price-only GRU conditions are excluded.

All scenarios now use same-day headline availability. **First-day event injection** sends an event only on its first eligible trading decision; **temporal event graph** repeats its decaying message over its active period. This distinction is separate from the source-date availability rule.

The real headlines expose limitations hidden by the synthetic examples: a dollar symbol does not establish USD, unknown counterparties cannot be assigned invented tickers, and analyst price-target changes are not management guidance. Original extraction records keep their review flags. `experiments/events.py` creates **separate partial records** with new IDs, the verbatim headline, surface-checked categories, and independently resolved quoted participants. It omits uncertain structured amounts, duration, occurrence date, specific participant roles and unresolved counterparties. Every omission and source event is traceable. These partial records are not human-reviewed or semantically certified. Exact structured financial facts remain available in the original extraction output for a later annotated audit.

Both temporal backbones use the same typed event and correlation message operators, with a five-session event half-life and a 20-session TTL. The compact TFT implements variable selection, four static contexts, LSTM encoder/decoder layers, static enrichment, shared-value causal attention, and gated residual paths. It adapts the original quantile objective to the same return/direction/log-volatility heads and loss as GRU. This is a TFT architecture adaptation, not a reproduction of the original paper's results. The implementation is in `stak_graph/models/tft.py`.

A one-hop direct event graph can also be expressed as weighted event fusion. Confidence stays uncalibrated, and the extractor supplies no stock-impact direction. The 2019 benchmark has already been examined in the prior experiment, so the new comparison is exploratory rather than confirmation on a fresh holdout.

Metrics include direction/balanced accuracy, AUC, Brier score, return RMSE, mean daily cross-sectional rank IC and log-volatility RMSE, on all test examples and a matched active-event subset. Paired 20-session moving-block bootstrap intervals keep all stocks from a date together (2,000 replicates); they are exploratory and are not corrected for multiple comparisons. This pilot does not evaluate trading profitability or costs.

### Run the study

In a suitable environment install `.[local,embeddings,experiment]`, then use fresh stage/output directories:

```bash
python -m stak_graph.experiments.data \
  --config configs/real_pilot_v3.json \
  --raw-root /tmp/stak-study/raw \
  --output /tmp/stak-study/prepared

python -m stak_graph.extraction extract \
  --input /tmp/stak-study/prepared/documents.jsonl \
  --registry /tmp/stak-study/prepared/companies.csv \
  --backend transformers --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --model-cache /tmp/stak-graph-models \
  --cache /tmp/stak-study/extraction.sqlite3 \
  --output /tmp/stak-study/extraction

python -m stak_graph.experiments.complete \
  --prepared /tmp/stak-study/prepared \
  --extraction /tmp/stak-study/extraction \
  --projected /tmp/stak-study/projected \
  --embeddings /tmp/stak-study/embeddings \
  --config configs/real_pilot_v3.json \
  --output results/my_real_study \
  --work /tmp/stak-study/work \
  --state /tmp/stak-study/state.json
```

The completion stage uses the A100-compatible GPU runtime and cached pinned model weights. Download EmbeddingGemma 2 into the supplied model cache first if absent. On this machine the separate temporary calendar dependencies are exposed with `PYTHONPATH=/tmp/stak-graph-python-deps:/home/mshanthappa/STAK-Graph`; the existing ML interpreter remains unchanged. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before GPU training for deterministic cuBLAS operations.

The output contains `results.html`, `summary.json`, the fixed `plan.json`, all 30 selected neural checkpoints and training histories, train-only scalers, per-seed and ensemble benchmark predictions, processed inputs, source/partial-event provenance, and checksums. `experiments/verify.py` reconstructs inputs and targets, reloads both primary backbones, recomputes metrics independently, checks future-input invariance, and confirms that excluded baselines are absent.

For an already-extracted source run, `experiments/same_day.py` retimes and revalidates the archived real-model responses and reuses exact matching semantic vectors. `experiments/finish_comparison.py` trains the selected variants, verifies saved predictions, and publishes the report with a compressed artifact archive. The current run uses `/tmp/stak-graph-same-day-v1` for large working files and `results/same_day_tft_v1` for the report. Archive placement and reconstruction instructions are recorded in `publication.json`; the uncompressed temporary directory can be regenerated from the archive.

A fixed large-company universe, rounded/uncertain news timestamps, retrospective price adjustments, sparse headline sampling, pretrained-model historical contamination, unannotated extraction quality, and a single temporal holdout limit the interpretation. Multiple walk-forward periods, historical/delisted security coverage, an annotated event audit, calibrated confidence and cost-aware economic evaluation remain later research work.

## Expanded public sources and model comparison

The completed, verified comparison is in
[`results/public_model_comparison_v2/results.html`](results/public_model_comparison_v2/results.html).
It contains 33 supervised checkpoints plus native Chronos-2 forecasts. The
typed-graph GRU reaches 53.87% accuracy; GAT + TimeMachine reaches 53.42%, and
GAT + GRU reaches 53.05%. The matched event-edge effect for GAT + GRU is +0.27
percentage points, with a paired 95% interval of −0.52 to +0.77 points. These
results do not demonstrate a reliable event-driven improvement.

The new acquisition modules are `stak_graph/acquisition/public.py` and
`stak_graph/acquisition/earnings_releases.py`. Public acquisition needs no paid
account. Set the authorized SEC contact in `SEC_CONTACT_EMAIL`; it is sent in
request headers and is not serialized in source code or output manifests.
Requests are throttled below SEC's maximum rate. Responses are cached with URLs,
retrieval times, sizes and SHA256 hashes, and changed cached responses are rejected.

The acquired dataset contains:

| Source | Acquired evidence | Historical use |
| --- | --- | --- |
| SEC | 1,728 filing metadata records: 1,472 8-K, 192 10-Q and 64 10-K; no 6-K for these domestic issuers in the target window | Dated filing evidence; quarterly events described below |
| SEC quarterly coverage | All 256 target fiscal stock-quarters, FY2016–FY2019: 219 directly reported quarterly diluted-EPS observations and 37 earnings-release excerpts | Available at the end of the SEC filing date, never at the financial period end |
| SEC + OpenFIGI | CIK/ticker/exchange data and FIGI responses for all 16 stocks | Current identifier snapshots are not a complete historical security master |
| GDELT | 2,735 unique article URL records, with organizations and themes, from 46 successfully parsed 15-minute historical archives | Discovery candidates; no claim of full historical news coverage, verified entity links, article text or extracted financial events |
| Wikidata | All 16 companies resolved by CIK, 273 parent/subsidiary/industry claims with revision IDs and source qualifiers/references | Current snapshots are retained for future work and excluded from the 2016–2019 graph |

The quarterly grid uses **fiscal** years and quarters. Some announcements lie
outside the calendar-year forecasting window and remain excluded until their
actual source-availability date. EPS is never obtained by subtracting annual and
nine-month EPS. The 37 release-based records establish an earnings announcement;
they do not fabricate a missing structured EPS value. The original headline
earnings releases and subsequent SEC filings may concern the same economic
quarter: counts refer to source observations, not necessarily independent events.

The expanded forecasting input has **381 source observations**: the previous
125 projected headline records, 219 structured SEC disclosures and 37 release
excerpts. The historical graph contains 365 eligible Event nodes, 16,448 StockDay
nodes and 7,372 direct event-to-day edges. Records outside the supplied calendar
or after the historical cutoff are excluded. GDELT discovery metadata and current
Wikidata claims are not silently turned into historical event edges.

`stak_graph/models/benchmark.py` adds two-layer multihead **GAT** and
**SAGN-style feature-hop attention**, with the existing typed-message model as a
reference. Edges include Event→StockDay, same-date rolling correlation and forward
temporal connections. No StockDay→Event reverse path or future-return label
propagation is introduced. SAGN's self-label enhancement is not used.

The supervised sweep holds GAT fixed while comparing GRU, compact TFT, xLSTM's
official all-mLSTM stack, official TimeMachine, and TS-Library PatchTST,
iTransformer and TimeMixer bodies. All use the same 30-session history,
train/validation/test examples, train-only scaling, three fixed seeds and
multitask readout. GAT/SAGN graph comparisons hold GRU fixed. A GAT model without
event edges provides a matched event ablation. Training-mean, linear-price and
price-only GRU remain excluded.

These are **forecasting-task adaptations**, not reproductions of the papers'
scores. The scratch temporal models operate on 32 learned graph channels;
iTransformer's variable tokens therefore represent learned channels. TFT also
receives the 18 physical historical features. TimeMachine uses the authors'
Mamba equations and official Torch reference scan because a CUDA SDK is absent.
An explicit loader changes upstream in-place divisions to mathematically
equivalent out-of-place operations so gradients can reach the graph encoder.
The original repositories remain unmodified and their commits are recorded.

Chronos-2 is evaluated in two separate protocols: its official frozen encoder
with a supervised multitask head, and native pretrained median forecasts of
stock/SPY open/close with historical event/price/correlation covariates. The
native score computes the same next-open-to-fifth-close excess-return direction;
it has no fine-tuning, future covariates or cross-task learning. Its pretraining
overlap with these old market prices is unknown. It is not a matched scratch
training comparison. GOFA's released graph-language implementation has been
assessed; it has no completed forecasting adaptation or accuracy result here.

The current implementation passes **148 tests**, including graph causality,
real-event edge removal, original-vintage quarterly selection, late SEC
acceptance handling, and input gradients through the upstream temporal models.
The 2019 test has been inspected in earlier work: all new results remain
exploratory, with calendar-block intervals and no original-paper reproduction,
uncontaminated zero-shot, trading-profitability or paper-readiness claim.

Large data and checkpoints stay under `/var/tmp/mshanthappa-stak-graph`, keeping
the nearly full home filesystem free. Acquisition can be resumed from cached
raw responses. The stage commands are:

```bash
python -m stak_graph.acquisition.public \
  --config configs/real_pilot_v3.json --output /var/tmp/mshanthappa-stak-graph/knowledge_v2
python -m stak_graph.acquisition.earnings_releases \
  --knowledge /var/tmp/mshanthappa-stak-graph/knowledge_v2
python -m stak_graph.experiments.expanded \
  --source /var/tmp/mshanthappa-stak-graph/same_day_tft_v1 \
  --knowledge /var/tmp/mshanthappa-stak-graph/knowledge_v2 \
  --output /var/tmp/mshanthappa-stak-graph/expanded_v3 --include-releases
python -m stak_graph.experiments.model_sweep \
  --source /var/tmp/mshanthappa-stak-graph/same_day_tft_v1 \
  --expanded /var/tmp/mshanthappa-stak-graph/expanded_v3 \
  --output /var/tmp/mshanthappa-stak-graph/model_sweep_v2
python -m stak_graph.experiments.verify_sweep \
  --source /var/tmp/mshanthappa-stak-graph/same_day_tft_v1 \
  --expanded /var/tmp/mshanthappa-stak-graph/expanded_v3 \
  --output /var/tmp/mshanthappa-stak-graph/model_sweep_v2
python -m stak_graph.experiments.chronos_native \
  --source /var/tmp/mshanthappa-stak-graph/same_day_tft_v1 \
  --expanded /var/tmp/mshanthappa-stak-graph/expanded_v3 \
  --output /var/tmp/mshanthappa-stak-graph/model_sweep_v2
python scripts/verify_native_forecast.py \
  --run /var/tmp/mshanthappa-stak-graph/model_sweep_v2
```

On this machine use `/var/tmp/mshanthappa-robotics/venv/bin/python` without
modifying that environment, expose isolated dependencies with
`PYTHONPATH=/tmp/stak-benchmark-deps:/home/mshanthappa/STAK-Graph`, and set
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Use fresh expanded-data and output directories when changing evidence or model
code; the training plan rejects changes during resume.
After downloading weights, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` keeps cached
Chronos evaluation independent of checkpoint-service availability.

For a separate environment, install `.[benchmark]`. Restore the exact upstream
commits and Chronos checkpoint with `scripts/fetch_benchmark_sources.py --run
/var/tmp/mshanthappa-stak-graph/model_sweep_v2 --external /path/to/external
--download-chronos`; set `STAK_EXTERNAL` to that directory. The fetcher verifies
existing checkouts rather than changing them to another commit. Model weights
are cached separately and are not bundled into the small project repository.

## Primary implementation references

- [Qwen3-4B model and non-thinking usage](https://huggingface.co/Qwen/Qwen3-4B)
- [XGrammar](https://github.com/mlc-ai/xgrammar)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Gemma 4 instruction models](https://ai.google.dev/gemma/docs/core/model_card_4)
- [EmbeddingGemma 2 model, task prefixes, pooling, and modality configuration](https://huggingface.co/google/embeddinggemma-2)
- [FNSPID paper](https://arxiv.org/abs/2402.06698) and [pinned public data release](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/bf9189c41527198897d1af3e17b1a0095279fc45)
- [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars)
- [Temporal Fusion Transformer paper](https://arxiv.org/html/1912.09363v3) and [authors' implementation](https://github.com/google-research/google-research/tree/master/tft)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [OpenFIGI API](https://www.openfigi.com/api/documentation), [GDELT data](https://www.gdeltproject.org/data.html), [Wikidata CIK identifiers](https://www.wikidata.org/wiki/Property:P5531)
- [GAT](https://arxiv.org/abs/1710.10903), [SAGN](https://arxiv.org/abs/2104.09376), [GOFA implementation](https://github.com/JiaruiFeng/GOFA)
- [xLSTM](https://github.com/NX-AI/xlstm), [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine), [Mamba](https://github.com/state-spaces/mamba)
- [TS-Library PatchTST, iTransformer and TimeMixer implementations](https://github.com/thuml/Time-Series-Library), [Chronos-2 implementation](https://github.com/amazon-science/chronos-forecasting)

## Expanded 100-stock, ten-year study

The authorized follow-up has two separately reported stages. Stage 1 expands
the observed panel to 100 companies over 2014–2023 and compares the existing
FinSTAK design, price-only, no-decay, LSTM and T-GCN controls. Stage 2 tests
the proposal components that were absent from the original pilot. The old
16-stock results are preserved and are not directly comparable across periods.

The configuration is `configs/finstak_100x10_v1.json`. It freezes sector quotas
and candidate order; selection eligibility uses pre-2014 price observations.
The manually curated universe still has selection/survivorship limitations.
Twenty incomplete FNSPID company histories and one invalid observation require
complete supplementary company series from Yahoo's public historical chart
endpoint. Original exports and every downloaded response are retained. The
pipeline never splices differently adjusted providers within a company series.
All prices remain retrospectively adjusted research data, not a reconstructed
point-in-time corporate-action database. The final source session is 2023-12-28.

Both pinned FNSPID news archives are required: `All_external.csv` ends in 2020;
`nasdaq_exteral_data.csv` supplies later coverage and original article bodies.
Accepted events retain source-row hashes, verbatim assertion spans, named-company
evidence and category checks. Repeated source articles are versioned conservatively;
near-duplicate observations are clustered separately. SEC facts and narrow declared
8-K items supply additional official disclosures, with later earnings observations
excluded when they plausibly repeat the same reporting episode. The final merged
collection must pass the 25,000–50,000-record count gate before full training.

**Verification means automated source-evidence checks.** It does not mean human
semantic certification, independent verification that news claims are true, exact
economic-event deduplication, or observed historical delivery time. News is delayed
two calendar days after its reported timestamp; filings are available at the end
of the later filing date or UTC acceptance date. Five late-acceptance cases in
this study required that correction. A separate pass rejoins every retained news
assertion to its raw source row and checks official accessions/items/quarterly
facts. The initial six event categories yielded 24,260 merged records. Explicit
dividend declarations are collected as a separate seventh category to close the
observed count shortfall without retaining rejected or duplicate observations.
Event-category embeddings and typed attention size themselves from the recorded
vocabulary; legacy six-category models retain their original defaults.

The new runner batches complete decision dates rather than recomputing the full
historical graph on every update. An equivalence test checks it against the old
FinSTAK implementation with identical weights and inputs. Training uses three
paired seeds, at most 25 epochs, validation patience five, and 16 dates per batch
(1,600 stock examples). Runtime plans record the authoritative training settings.
Each annual fold refits its training-only scalers: train/validation end in
2019/2020 for the 2021 test, 2020/2021 for 2022 and 2021/2022 for 2023. Labels
crossing a split boundary are purged. Every model's checkpoint is selected by
validation normalized OHLC MAE before its fold's test scoring.

Stage 2 includes isolated wick heads, OHLCV targets, PCGrad, uncertainty weighting,
bounded exponential/power-law decay, explicit sector aggregation, a conservative
primary-subject filter, two-layer GRUs, a separate auxiliary direction head,
frozen FinBERT embeddings and disjoint valuation/dispersion streams. The combined
recipe uses 20 sessions, a two-layer GRU, FinBERT, hybrid decay, sector routing,
primary-subject filtering, detached wick features and boundaries, OHLCV targets,
an auxiliary direction task and PCGrad. These are custom project implementations,
not a full published HGT reproduction. Historical supply-chain edges, consensus
surprises, court-docket diffs and a verified pre-market feed are not fabricated.

Volume retransformation compares naive exponentiation, per-stock Gaussian
residual correction and empirical smearing, calibrated only on the preceding
validation year. Reported MSE and MAPE can favor different corrections. A frozen,
prespecified 128-session TimesFM price-only comparator uses the same panel;
foundation pretraining overlap remains unknown.

The tracked entry point is `scripts/run_scale_pipeline.py`. It waits for acquisition,
freezes implementation sources, runs tests and independent source checks, creates
real frozen text embeddings, runs Stage 1 and then Stage 2, verifies saved labels
and checkpoint reconstruction, perturbs future inputs, and builds the result report.
Check `results/scale_100x10_v1/pipeline_status.json` for actual progress. A count
shortfall or failed audit stops the full study rather than silently weakening a
requirement. Completed results belong in that directory's `results.html` and
`summary.json`; the existence of code or a GPU smoke check is not a training result.

### Completed expanded-study results

The completed collection contains 100 stocks, 2014–2023 observations and 27,030
automatically source-evidence-checked event records. There are no human-certified
events. Both experiment stages finished: 17 configurations, three paired seeds
and three annual test folds, totaling 153 fits and 74,900 held-out stock-date
forecasts per configuration. The frozen implementation passed 183 unit tests.
Independent checks hashed all 153 checkpoints and reconstructed 51 seed-7
model/year forecasts. Future-input and target-label perturbations changed none
of those earlier model outputs.

Pooled normalized OHLC MAE was 0.965918% for the price-only recurrent control and
0.964184% for no-decay FinSTAK, a small 0.18% relative error reduction. Hybrid
decay reached 0.964000%, but its exploratory paired interval against no-decay
FinSTAK includes zero. The combined recipe reached 0.973310%; frozen native
TimesFM reached 1.219086%. The new-component family, with each year's recipe
selected only on validation data, reached 0.964994% versus 0.964184% for the
validation-selected original family. This study does not establish a reliable
price-forecasting improvement from the added components. The latest-fold
validation choice remains no-decay FinSTAK.

The retained report, outcome JSON and research archive are in
`results/scale_100x10_v1_retained/`. The archive contains frozen source, fitted
checkpoints, packed inputs, predictions and audit evidence. Pretrained model
weights and the 32 GB original news exports are excluded; historical replay uses
retained embeddings. A fresh-directory CPU replay from the archive passed.

The initial verification used a different inference batch size; the corrected
checker uses the original 16-date batch and the unchanged error tolerance.
Saved seed-7 forecasts then reproduced to rounding precision, while independent
two-date causality checks still passed. A report-only dictionary variable was
renamed to avoid shadowing a metric function. Neither correction changed the
frozen training implementation, checkpoints or saved predictions. Original and
revised postprocessing code and provenance are retained with the results.
