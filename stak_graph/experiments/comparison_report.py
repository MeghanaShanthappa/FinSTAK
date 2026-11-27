"""Current same-day TFT/GRU report; only explicitly selected graph conditions."""

from html import escape
import json
from pathlib import Path

from stak_graph.io import write_json
from .data import sha256_file

SCENARIO_LABELS = {
    "price_correlation": "Price + correlation graph",
    "same_day_event": "Same-day event injection",
    "event_graph": "Temporal event graph",
    "event_graph_correlation": "Event + correlation graph",
    "historical_event_control": "Historical-content control",
}


def label(name):
    stem, architecture = name.rsplit("_", 1)
    return SCENARIO_LABELS[stem] + " " + architecture.upper()


def pct(value): return f"{100*value:.2f}%"
def num(value): return "—" if value is None else f"{value:.4f}"


def make_report(output, project):
    output, project = Path(output), Path(project)
    summary = json.loads((output / "summary.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    primary_name = "__vs__".join(summary["primary_comparison"])
    primary = summary["paired_comparisons"][primary_name]["direction_accuracy_gain"]
    lo, hi = primary["ci95"]
    if lo > 0: finding = "TFT improved direction accuracy relative to GRU in this same-day benchmark comparison."
    elif hi < 0: finding = "TFT reduced direction accuracy relative to GRU in this same-day benchmark comparison."
    else: finding = "This comparison does not establish a reliable direction-accuracy improvement from TFT over GRU."
    comparisons = []
    for stem, name in SCENARIO_LABELS.items():
        gru, tft = stem + "_gru", stem + "_tft"
        if gru not in summary["models"] or tft not in summary["models"]: continue
        gain = summary["paired_comparisons"][tft + "__vs__" + gru]["direction_accuracy_gain"]
        comparisons.append(f"<tr><th>{escape(name)}</th><td>{pct(summary['models'][gru]['test']['direction_accuracy'])}</td>"
                           f"<td>{pct(summary['models'][tft]['test']['direction_accuracy'])}</td>"
                           f"<td>{100*gain['estimate']:+.2f} pp</td><td>[{100*gain['ci95'][0]:+.2f}, {100*gain['ci95'][1]:+.2f}]</td></tr>")
    tables = []
    for subset, title in (("test", "All 2019 benchmark stock days"), ("test_event_active_days", "Matched days with active event edges")):
        rows = []
        for architecture in ("gru", "tft"):
            for stem in SCENARIO_LABELS:
                name = stem + "_" + architecture
                if name not in summary["models"]: continue
                model = summary["models"][name]
                if subset not in model: continue
                value = model[subset]
                ci = model["test_accuracy_interval"]["ci95"] if subset == "test" else None
                interval = f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]" if ci else "—"
                seed = f"{100*model['seed_accuracy_mean']:.2f} ± {100*model['seed_accuracy_std']:.2f}"
                rows.append(f"<tr><th>{escape(label(name))}</th><td>{pct(value['direction_accuracy'])}</td><td>{num(value['balanced_accuracy'])}</td>"
                            f"<td>{num(value['roc_auc'])}</td><td>{num(value['mean_daily_rank_ic'])}</td><td>{num(value['excess_return_rmse'])}</td>"
                            f"<td>{num(value['log_volatility_rmse'])}</td><td>{seed}</td><td>{interval}</td></tr>")
        tables.append(f"<section><h2>{title}</h2><div class='scroll'><table><thead><tr><th>Condition</th><th>Ensemble direction accuracy</th>"
                      "<th>Balanced accuracy</th><th>AUC</th><th>Daily rank IC</th><th>Return RMSE ↓</th><th>Log-volatility RMSE ↓</th>"
                      f"<th>Run accuracies (%), mean ± SD</th><th>Accuracy 95% interval (%)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>")
    primary_candidate, primary_baseline = summary["primary_comparison"]
    split = summary["split_counts"]
    all_pair_rows = []
    for key, result in summary["paired_comparisons"].items():
        candidate, baseline = key.split("__vs__")
        accuracy = result["direction_accuracy_gain"]; rank = result["daily_rank_ic_gain"]
        all_pair_rows.append(f"<tr><th>{escape(label(candidate))}<br><small>vs {escape(label(baseline))}</small></th>"
                             f"<td>{100*accuracy['estimate']:+.2f} pp [{100*accuracy['ci95'][0]:+.2f}, {100*accuracy['ci95'][1]:+.2f}]</td>"
                             f"<td>{num(rank['estimate']) if rank else '—'}</td></tr>")
    limits = summary["data_limitations"] + summary["method_limitations"]
    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STAK-Graph — same-day TFT and GRU comparison</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f6fa;color:#182433;line-height:1.55;max-width:1400px;margin:auto;padding:28px}}
h1{{line-height:1.2}}h2{{font-size:1.25rem}}section,.finding{{background:white;border:1px solid #dce3ed;border-radius:10px;padding:22px;margin:22px 0}}
.finding{{border-left:6px solid #365eaa}}small,.muted{{color:#526174}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:11px;border-bottom:1px solid #e5e9f0;text-align:right;vertical-align:top}}th:first-child{{text-align:left;min-width:230px}}
thead{{background:#edf1f7}}thead th{{white-space:nowrap}}a{{color:#254f9a}}li{{margin:8px 0}}code{{background:#edf1f7;padding:2px 5px}}
</style></head><body><p class="muted">STAK-Graph · same-day recorded-date news · matched graph conditions</p>
<h1>TFT versus GRU with same-day financial events</h1>
<div class="finding"><strong>{escape(finding)}</strong><p>Primary pair: {escape(label(primary_candidate))} versus {escape(label(primary_baseline))}.
Direction difference: <strong>{100*primary['estimate']:+.2f} percentage points</strong>; paired 95% interval <strong>[{100*lo:+.2f}, {100*hi:+.2f}]</strong>.</p>
<p>The previously examined 2019 benchmark is reused. These results are exploratory and do not establish paper-worthy improvement on a fresh holdout.</p></div>
<section><h2>Matched architecture comparison</h2><p>Both columns use identical source dates, event text vectors, stock prices, targets, splits and graph rules.
Each neural result combines predicted probabilities from three seeds. The models read 30 trading sessions and predict five-session market-adjusted return direction.</p>
<div class="scroll"><table><thead><tr><th>Graph condition</th><th>Same-day GRU</th><th>Same-day TFT</th><th>TFT − GRU</th><th>Paired 95% interval</th></tr></thead>
<tbody>{''.join(comparisons)}</tbody></table></div></section>
<section><h2>Same-day availability</h2><p>All {summary['source_news_documents']} source headlines have date-only midnight placeholders.
They are assigned <strong>23:59:59 UTC on their recorded date</strong>. Forecast decisions occur at that same end-of-day cutoff on trading days,
and the holding period starts at the next trading open. Headlines first available on weekends or holidays enter the next supplied trading decision.
This is a recorded-date assumption; historical arrival times and the accuracy of vendor source dates are unverified.</p>
<p>Prices remain available 15 minutes after the actual market close. A headline from a later date cannot enter an earlier decision.
The two-calendar-day delay has been removed. Frozen model responses and vectors were reused after evidence and identity revalidation, with full provenance.</p></section>
<section><h2>Architecture and evaluation</h2><p>GRU uses the shared graph message operators followed by a forward-only recurrent encoder.
The compact TFT uses the same graph operators, separate historical-variable selection, static company context, an LSTM encoder/decoder,
static enrichment, shared-value causal attention, and gated residual paths. Its only known-future variable is the relative forecast-horizon index;
future prices, future news and labels are never decoder inputs. The original TFT quantile objective is adapted to the same return, binary-direction,
and log-volatility objectives as GRU. This is an implementation following the published architecture, not a reproduction of the paper's results.</p>
<p>Train 2016–2017: {split['train']:,} examples. Validation 2018: {split['validation']:,}. Benchmark 2019: {split['test']:,} across 16 stocks.
There are {split['purged']:,} purged examples. Feature/target scaling and gradient updates use training data only.
Checkpoints are selected by validation multitask loss. All selected conditions finish fitting before new benchmark scores are computed.</p>
<p>Intervals use 2,000 moving-block resamples of 20 trading sessions, keeping all stocks from each date together. Secondary comparisons are exploratory.</p></section>
{''.join(tables)}
<section><h2>Additional paired comparisons</h2><div class="scroll"><table><thead><tr><th>Candidate and comparison</th><th>Direction gain and interval</th><th>Daily rank-IC gain</th></tr></thead>
<tbody>{''.join(all_pair_rows)}</tbody></table></div></section>
<section><h2>Scope and limitations</h2><p>{summary['graph_statistics']['events']} source-traceable partial events feed {summary['graph_statistics']['direct_active_edges']:,} direct active edges.
{pct(summary['test_event_active_fraction'])} of benchmark examples have active event edges. Graph weights decay with a five-session half-life and stop new injection at 20 sessions.
Confidence remains uncalibrated; semantic correctness has not been established by a human-labeled extraction audit.</p>
<ul>{''.join('<li>'+escape(value)+'</li>' for value in limits)}</ul></section>
<section><h2>Saved artifacts</h2><p><a href="summary.json">Metrics</a> · <a href="plan.json">Fixed experiment plan</a> ·
<a href="verification.json">Independent reconstruction checks</a> · <a href="artifact_hashes.json">Checksums</a> · <a href="test_samples.jsonl">Benchmark timestamps</a>.
Selected checkpoints, histories, processed data, cached-response provenance and predictions are retained with the experiment.</p>
<p>Sources: <a href="https://arxiv.org/html/1912.09363v3">TFT paper</a>, <a href="https://github.com/google-research/google-research/tree/master/tft">authors' reference implementation</a>,
<a href="https://huggingface.co/datasets/Zihan1004/FNSPID">FNSPID</a>.</p><p class="muted">Experiment: <code>{escape(summary['experiment_id'])}</code>.
Numbers are generated from saved predictions; removed baseline conditions are absent from this experiment.</p></section></body></html>"""
    with (output / "results.html").open("x", encoding="utf-8") as target: target.write(content)
    hashes = {str(path.relative_to(output)): sha256_file(path) for path in output.rglob("*") if path.is_file()}
    source_hashes = {str(path.relative_to(project)): sha256_file(path) for base in (project / "stak_graph", project / "configs")
                     for path in base.rglob("*") if path.is_file() and path.suffix in (".py", ".json", ".txt")}
    write_json(output / "artifact_hashes.json", {"artifacts": hashes, "source_code": source_hashes})
    return {"report": str((output / "results.html").resolve()), "finding": finding}
