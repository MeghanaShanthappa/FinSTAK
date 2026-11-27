"""Generate a standalone, reproducible results view from saved test predictions."""

import argparse
from html import escape
import json
from pathlib import Path

from stak_graph.io import write_json
from .data import sha256_file

LABELS = {
    "training_mean": "Training-mean / majority baseline",
    "linear_price": "Linear price model",
    "price_gru": "Price-only GRU",
    "price_correlation_gru": "Price + correlation graph GRU",
    "same_day_event_gru": "Price + same-day event GRU",
    "event_graph_gru": "Temporal event graph GRU",
    "event_graph_correlation_gru": "Event + correlation graph GRU",
    "historical_event_control_gru": "Historical-content negative control",
}


def percent(value):
    return f"{100*value:.2f}%"


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(output, project):
    output, project = Path(output), Path(project)
    summary = json.loads((output / "summary.json").read_text())
    if "primary_comparison" in summary:
        from .comparison_report import make_report as comparison_report
        return comparison_report(output, project)
    plan = json.loads((output / "plan.json").read_text())
    primary = summary["paired_comparisons"]["event_graph_correlation_gru__vs__price_correlation_gru"]
    direction = primary["direction_accuracy_gain"]
    low, high = direction["ci95"]
    if low > 0:
        finding = "The event graph improved direction accuracy in this pilot; broader validation is still required."
    elif high < 0:
        finding = "The event graph reduced direction accuracy relative to the correlation baseline in this pilot."
    else:
        finding = "This pilot does not establish a direction-forecasting improvement from the event graph."
    tables = []
    for subset, title in (("test", "All held-out stock days"), ("test_event_active_days", "Held-out days with an active event edge")):
        rows = []
        for name in LABELS:
            model = summary["models"][name]
            if subset not in model: continue
            metrics = model[subset]
            variation = "—"
            if "seed_accuracy_std" in model:
                variation = f"{100*model['seed_accuracy_mean']:.2f} ± {100*model['seed_accuracy_std']:.2f}"
            interval = model.get("test_accuracy_interval", {}).get("ci95") if subset == "test" else None
            ci = f"[{100*interval[0]:.2f}, {100*interval[1]:.2f}]" if interval else "—"
            rows.append(f"<tr><th scope='row'>{escape(LABELS[name])}</th><td>{percent(metrics['direction_accuracy'])}</td>"
                        f"<td>{number(metrics['balanced_accuracy'])}</td><td>{number(metrics['roc_auc'])}</td>"
                        f"<td>{number(metrics['mean_daily_rank_ic'])}</td><td>{number(metrics['excess_return_rmse'])}</td>"
                        f"<td>{number(metrics['log_volatility_rmse'])}</td><td>{variation}</td><td>{ci}</td></tr>")
        tables.append(f"<section><h2>{title}</h2><div class='scroll'><table><thead><tr><th>Model</th><th>Direction accuracy</th>"
                      "<th>Balanced accuracy</th><th>AUC</th><th>Daily rank IC</th><th>Excess-return RMSE ↓</th>"
                      "<th>Log-volatility RMSE ↓</th><th>Seed accuracy mean ± SD (pp)</th><th>Accuracy 95% interval (%)</th>"
                      f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>")
    comparisons = []
    for label, result in summary["paired_comparisons"].items():
        candidate, baseline = label.split("__vs__")
        accuracy = result["direction_accuracy_gain"]
        rank = result["daily_rank_ic_gain"]
        vol = result["log_volatility_mse_reduction"]
        comparisons.append(f"<tr><th scope='row'>{escape(LABELS[candidate])}<br><small>vs {escape(LABELS[baseline])}</small></th>"
                           f"<td>{100*accuracy['estimate']:+.2f} pp<br><small>[{100*accuracy['ci95'][0]:+.2f}, {100*accuracy['ci95'][1]:+.2f}]</small></td>"
                           f"<td>{number(rank['estimate']) if rank else '—'}<br><small>{escape(str(rank['ci95'])) if rank else ''}</small></td>"
                           f"<td>{number(vol['estimate'])}<br><small>{escape(str(vol['ci95']))}</small></td></tr>")
    limitations = summary["data_limitations"] + summary["method_limitations"]
    split = summary["split_counts"]
    projection = summary.get("event_projection") or {}
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STAK-Graph — real-data pilot results</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#172333;background:#f4f6fa;line-height:1.55}}
body{{max-width:1350px;margin:0 auto;padding:32px}}h1{{font-size:2.1rem;line-height:1.15}}h2{{font-size:1.2rem}}
section,.finding{{background:white;border:1px solid #dce2eb;border-radius:12px;padding:22px;margin:20px 0}}
.finding{{border-left:6px solid #4265ad}}.muted,small{{color:#526174}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:11px;text-align:right;border-bottom:1px solid #e5e9f0;vertical-align:top}}
th:first-child{{text-align:left;min-width:220px}}thead th{{white-space:nowrap;background:#edf1f7}}a{{color:#254f9a}}
code{{background:#edf1f7;padding:2px 5px;border-radius:3px}}li{{margin:8px 0}}.stats{{display:flex;gap:18px;flex-wrap:wrap}}
.stats span{{padding:10px 16px;background:#eaf0f9;border-radius:8px}}@media print{{body{{padding:0;background:white}}section{{break-inside:avoid}}}}
</style></head><body>
<p class="muted">STAK-Graph · reproducible retrospective pilot · real prices and real headlines</p>
<h1>Do temporal financial-event edges improve forecasting?</h1>
<div class="finding"><strong>{escape(finding)}</strong><p>Primary comparison: event + correlation graph GRU against the price + correlation graph GRU.
Direction accuracy difference: <strong>{100*direction['estimate']:+.2f} percentage points</strong>; paired 95% interval
<strong>[{100*low:+.2f}, {100*high:+.2f}]</strong>. A paper-worthy improvement is not established by this single pilot.</p></div>
<div class="stats"><span>16 stocks</span><span>{summary['source_news_documents']} headlines</span>
<span>{projection.get('projected_events', summary['graph_statistics']['events'])} partial event records</span>
<span>{split['test']:,} test examples</span><span>3 seeds per neural condition</span><span>{escape(summary['gpu'] or summary['device'])}</span></div>
<section><h2>Study design</h2>
<p>Train: 2016–2017 ({split['train']:,} examples). Validation: 2018 ({split['validation']:,}). Test: 2019 ({split['test']:,}).
The {split['purged']:,} examples whose five-session labels cross a training/validation boundary are excluded from fitting and model selection.</p>
<p>At each close plus 15 minutes, the model reads the preceding 30 sessions. It predicts the stock's log return minus SPY's log return,
from the next session's open to the fifth following session's close. The volatility target is the log RMS of daily holding-period returns.</p>
<p>Headlines are sampled by fixed hashes: one general headline and, where available, one event-keyword headline per company/month.
Every headline becomes available two calendar days after its source date. This is an assumption because verified historical arrival times are absent.
Feature and target scaling uses training data only. Checkpoints use validation loss only; test metrics are computed after all training runs finish.</p>
<p>All neural conditions receive the same learned company-identity embeddings; the linear model receives company indicators.
This gives price baselines access to company identity even when their inputs contain no event text.</p>
<p>Neural results show the mean prediction across three seeds. Seed statistics describe individual runs. Intervals resample 20-session calendar blocks
with all stocks on each date kept together (2,000 replicates). They are exploratory and not corrected for multiple comparisons.</p></section>
{''.join(tables)}
<section><h2>Paired comparisons</h2><p>Positive values favor the event candidate. Brackets give paired 95% block-bootstrap intervals.</p>
<div class="scroll"><table><thead><tr><th>Candidate and baseline</th><th>Direction gain</th><th>Daily rank-IC gain</th><th>Log-volatility MSE reduction</th></tr></thead>
<tbody>{''.join(comparisons)}</tbody></table></div></section>
<section><h2>What the event graph tested</h2>
<p>Event messages use frozen EmbeddingGemma 2 vectors, coarse categories and typed event-to-day edges. Their fixed weights decay with a five-session half-life
and stop new injection after 20 sessions. Confidence remains uncalibrated. The correlation baseline uses up to three positive neighbors from the preceding
60 sessions. Same-day fusion sees the same event content on its first eligible session. The negative control replaces content only with already-known historical events.</p>
<p>Original extraction flags remain intact. Separate partial event records preserve the exact headline and quoted, resolved companies; uncertain amounts,
durations, dates, specific roles, and unresolved counterparties are omitted. Surface checks reject obvious category mismatches but do not establish semantic accuracy.
The single event-to-day message step can also be written as weighted event fusion, so a gain would support the temporal edge policy, rather than a general GNN advantage.</p>
<p>Active event edges cover {percent(summary['test_event_active_fraction'])} of test examples. The graph contains {summary['multi_company_events']} multi-company events,
{summary['graph_statistics']['propagated_active_edges']} business-relation exposure edges, and {summary['correlation_edges']:,} rolling-correlation edges.</p></section>
<section><h2>Limits on the conclusion</h2><ul>{''.join('<li>'+escape(item)+'</li>' for item in limitations)}</ul></section>
<section><h2>Reproducibility and sources</h2><p>Saved locally: <a href="summary.json">complete metrics</a>, <a href="plan.json">locked study plan</a>,
<a href="test_samples.jsonl">test timestamps and identities</a>, <a href="scaler.json">training-only scaler</a>,
<a href="linear_price_model.json">linear coefficients</a>, <a href="artifact_hashes.json">artifact hashes</a>.
<a href="verification.json">Independent reconstruction checks</a> reproduce saved targets, model predictions, and metrics.
Checkpoints, all seed histories, and compressed test predictions are retained beside this report.</p>
<p>Data: <a href="https://huggingface.co/datasets/Zihan1004/FNSPID">FNSPID release</a>,
<a href="https://arxiv.org/abs/2402.06698">FNSPID paper</a>.
Extraction: <a href="https://huggingface.co/Qwen/Qwen3-4B">Qwen3-4B</a>.
Embeddings: <a href="https://huggingface.co/google/embeddinggemma-2">EmbeddingGemma 2</a>.
Calendar: <a href="https://github.com/gerrymanoim/exchange_calendars">exchange_calendars</a>.</p>
<p class="muted">Experiment ID: <code>{escape(summary['experiment_id'])}</code>. Config hash: <code>{escape(summary['config_sha256'])}</code>.
Generated only from saved evaluation artifacts; no manually edited metric values.</p></section></body></html>"""
    with (output / "results.html").open("x", encoding="utf-8") as target:
        target.write(content)
    hashes = {str(path.relative_to(output)): sha256_file(path) for path in output.rglob("*") if path.is_file()}
    code_hashes = {str(path.relative_to(project)): sha256_file(path)
                   for directory in (project / "stak_graph", project / "configs")
                   for path in directory.rglob("*") if path.is_file() and path.suffix in (".py", ".json", ".txt")}
    write_json(output / "artifact_hashes.json", {"artifacts": hashes, "source_code": code_hashes})
    return {"report": str((output / "results.html").resolve()), "finding": finding}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(make_report(args.output, args.project), indent=2))


if __name__ == "__main__":
    main()
