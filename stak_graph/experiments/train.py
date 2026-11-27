"""Matched GPU baselines with validation-only checkpoint selection.

The test metrics are computed once, after all checkpoints have been selected.
This is an exploratory retrospective pilot, not a live trading backtest.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stak_graph.embeddings.pipeline import load_vectors
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.extraction.schema import digest
from stak_graph.graph.builder import EventPolicy, GraphBuilder
from stak_graph.graph.market import TradingCalendar, read_bars
from stak_graph.io import read_jsonl, write_json, write_jsonl
from stak_graph.models.event_gru import EventGraphGRU, tensorize
from .data import sha256_file
from .dataset import build_samples, causal_replacement, correlation_edges
from .metrics import block_interval, evaluate, paired_comparison

SCENARIOS = ("price_correlation", "same_day_event", "event_graph", "event_graph_correlation", "historical_event_control")
VARIANTS = tuple(f"{scenario}_{architecture}" for architecture in ("gru", "tft") for scenario in SCENARIOS)
SUPPORTED_GRU_VARIANTS = ("price_gru",) + tuple(f"{scenario}_gru" for scenario in SCENARIOS)
EXCLUDED_MODELS = ("training_mean", "linear_price", "price_gru")
LOSS_WEIGHTS = {"return_huber": 0.5, "direction_bce": 0.5, "log_volatility_mse": 0.2}


class WindowForecastModel(EventGraphGRU):
    """Same encoders/heads across conditions, with explicit typed messages.

    A single event-to-day message step followed by a causal, 30-session GRU.
    A one-hop direct event graph is equivalent to weighted event fusion; this
    experiment tests the edge/time policy, not a universal GNN advantage.
    """

    def __init__(self, price_dimension, event_dimension, hidden, variant, company_count=16):
        super().__init__(price_dimension, event_dimension, hidden)
        if variant not in SUPPORTED_GRU_VARIANTS:
            raise ValueError("Unknown experiment variant")
        self.variant = variant
        self.correlation_message = nn.Linear(hidden, hidden, bias=False)
        self.company_embedding = nn.Embedding(company_count, hidden)
        self.dropout = nn.Dropout(0.1)

    def graph_channels(self, data):
        """Identical graph operators for both temporal backbones."""
        prices = self.price_encoder(data["prices"])
        if "company_index" in data:
            prices = prices + self.company_embedding(data["company_index"])
        event_message = torch.zeros_like(prices)
        correlation_message = torch.zeros_like(prices)
        if self.variant not in ("price_gru", "price_correlation_gru"):
            events = self.event_encoder(data["events"]) + self.category_embedding(data["categories"])
            aggregate = torch.zeros_like(prices)
            count = torch.zeros((len(prices), 1), device=prices.device)
            for relation, layer in enumerate(self.relation_messages):
                selected = data["edge_relation"] == relation
                if self.variant == "same_day_event_gru":
                    selected = selected & (data["edge_age"] == 0) & (data["edge_relation"] == 0)
                target = data["edge_day"][selected]
                if target.numel():
                    messages = layer(events[data["edge_event"][selected]]) * data["edge_weight"][selected, None]
                    aggregate.index_add_(0, target, messages)
                    count.index_add_(0, target, torch.ones((len(target), 1), device=prices.device))
            event_message = aggregate / count.clamp_min(1).sqrt()
        if self.variant in ("price_correlation_gru", "event_graph_correlation_gru"):
            aggregate = torch.zeros_like(prices)
            count = torch.zeros((len(prices), 1), device=prices.device)
            target, source = data["corr_target"], data["corr_source"]
            if target.numel():
                aggregate.index_add_(0, target, self.correlation_message(prices[source]) * data["corr_weight"][:, None])
                count.index_add_(0, target, torch.ones((len(target), 1), device=prices.device))
            correlation_message = aggregate / count.clamp_min(1).sqrt()
        return prices, event_message, correlation_message

    def encode_days(self, data):
        prices, event_message, correlation_message = self.graph_channels(data)
        return self.fusion(prices + event_message + correlation_message)

    def predict_windows(self, daily_states, windows):
        sequence = self.dropout(daily_states[windows])
        encoded, _ = self.temporal(sequence)
        state = encoded[:, -1]
        return torch.stack((self.return_head(state)[:, 0], self.direction_head(state)[:, 0],
                            self.volatility_head(state)[:, 0]), dim=1)

    def forward(self, data, windows):
        return self.predict_windows(self.encode_days(data), windows)


def make_model(variant, price_dimension, event_dimension, hidden, company_count, horizon=5):
    if variant.endswith("_tft"):
        from stak_graph.models.tft import WindowTFTModel
        return WindowTFTModel(price_dimension, event_dimension, hidden, variant, company_count, horizon)
    return WindowForecastModel(price_dimension, event_dimension, hidden, variant, company_count)


def multitask_loss(prediction, target):
    return (LOSS_WEIGHTS["return_huber"] * F.smooth_l1_loss(prediction[:, 0], target[:, 0])
            + LOSS_WEIGHTS["direction_bce"] * F.binary_cross_entropy_with_logits(prediction[:, 1], target[:, 1])
            + LOSS_WEIGHTS["log_volatility_mse"] * F.mse_loss(prediction[:, 2], target[:, 2]))


def make_tensor_data(snapshot, vectors, dimension, study, device, config):
    data = tensorize(snapshot, vectors, dimension, device)
    data["prices"] = torch.as_tensor(study["features"], device=device)
    identifiers = sorted(data["sequences"])
    data["company_ids"] = identifiers
    data["company_index"] = torch.tensor([identifiers.index(node["company_id"]) for node in snapshot["nodes"]["stock_day"]],
                                         dtype=torch.long, device=device)
    data["edge_age"] = torch.tensor([edge["age_sessions"] for edge in snapshot["edges"] if edge["type"] == "ACTIVE_ON"],
                                    dtype=torch.long, device=device)
    source, target, weight = correlation_edges(snapshot, window=60, neighbors=3, minimum=0.2)
    data["corr_source"] = torch.as_tensor(source, device=device)
    data["corr_target"] = torch.as_tensor(target, device=device)
    data["corr_weight"] = torch.as_tensor(weight, device=device)
    return data


def standardized_targets(study, device):
    values = study["targets"].copy()
    values[:, 0] /= study["scaler"]["return_scale"]
    values[:, 2] = (values[:, 2] - study["scaler"]["log_vol_mean"]) / study["scaler"]["log_vol_scale"]
    return torch.as_tensor(values, device=device)


def restore_predictions(raw, scaler):
    values = np.asarray(raw, dtype=np.float64).copy()
    values[:, 0] *= scaler["return_scale"]
    values[:, 1] = 1 / (1 + np.exp(-values[:, 1].clip(-40, 40)))
    values[:, 2] = values[:, 2] * scaler["log_vol_scale"] + scaler["log_vol_mean"]
    return values


@torch.no_grad()
def predict(model, data, windows, indices, batch_size=1024):
    model.eval()
    daily = model.encode_days(data)
    return torch.cat([model.predict_windows(daily, windows[group]).cpu()
                      for group in indices.split(batch_size)]).numpy()


def fit_one(variant, seed, data, windows, targets, study, config, output):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(variant, data["prices"].shape[1], data["events"].shape[1], config["hidden_dimension"],
                       len(data["company_ids"]), config["horizon_sessions"]).to(data["prices"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    training = torch.as_tensor(np.flatnonzero(study["masks"]["train"]), device=windows.device)
    validation = torch.as_tensor(np.flatnonzero(study["masks"]["validation"]), device=windows.device)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch = float("inf"), None, 0
    history, start = [], time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = training[torch.randperm(len(training), generator=generator).to(training.device)]
        losses = []
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data, windows[indices])
            loss = multitask_loss(prediction, targets[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = torch.as_tensor(predict(model, data, windows, validation), device=windows.device)
        validation_loss = float(multitask_loss(validation_prediction, targets[validation]))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best - 1e-5:
            best, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"variant": variant, "seed": seed, **history[-1], "best_epoch": best_epoch}), flush=True)
        if epoch - best_epoch >= config["early_stopping_patience"]:
            break
    if best_state is None:
        raise ValueError("No valid checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / f"{variant}_seed{seed}.pt"
    torch.save({"state_dict": best_state, "variant": variant, "seed": seed, "hidden": config["hidden_dimension"],
                "price_dimension": data["prices"].shape[1], "event_dimension": data["events"].shape[1],
                "company_ids": data["company_ids"],
                "horizon": config["horizon_sessions"], "architecture": "tft" if variant.endswith("_tft") else "gru",
                "scaler": study["scaler"], "config_sha256": digest(config), "selected_epoch": best_epoch}, checkpoint)
    summary = {"variant": variant, "seed": seed, "selected_epoch": best_epoch, "epochs_run": len(history),
               "validation_loss": best, "elapsed_seconds": round(time.monotonic() - start, 3),
               "parameters_total_including_inactive_controls": sum(p.numel() for p in model.parameters()),
               "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(output / "histories" / f"{variant}_seed{seed}.json", {"summary": summary, "epochs": history})
    return model, summary


def run(prepared, extraction, embeddings, config, output, work, device, projected=None):
    started = time.monotonic()
    prepared, extraction, embeddings, output, work = map(Path, (prepared, extraction, embeddings, output, work))
    if output.exists() or work.exists():
        raise ValueError("Experiment output/work directory already exists")
    variants = tuple(config.get("variants", VARIANTS))
    if not variants or len(set(variants)) != len(variants) or any(name not in VARIANTS for name in variants):
        raise ValueError("Select unique retained graph variants; removed baselines cannot be run")
    if not (extraction / "manifest.json").is_file():
        raise ValueError("Extraction must finish before the forecasting experiment")
    extraction_manifest = json.loads((extraction / "manifest.json").read_text())
    if extraction_manifest["run_status"] == "aborted" or extraction_manifest["extraction"]["generator"].get("simulated"):
        raise ValueError("A real completed extraction run is required")
    if extraction_manifest["summary"]["document_errors"]:
        raise ValueError("Resolve source input errors before training")
    if extraction_manifest["processed_input_sha256"] != sha256_file(prepared / "documents.jsonl"):
        raise ValueError("Extraction input does not match the prepared real-data sample")
    if extraction_manifest["registry_sha256"] != sha256_file(prepared / "companies.csv"):
        raise ValueError("Extraction company registry does not match prepared data")
    if config != json.loads((prepared / "config.json").read_text()):
        raise ValueError("Training config differs from the recorded data sampling config")
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    for name, expected in prepared_manifest["files"].items():
        if sha256_file(prepared / name) != expected:
            raise ValueError(f"Prepared input checksum mismatch: {name}")
    vectors, embedding_manifest = load_vectors(embeddings)
    if embedding_manifest["encoder"].get("simulated"):
        raise ValueError("Real semantic embeddings are required")
    calendar = TradingCalendar.from_csv(prepared / "calendar.csv")
    bars = read_bars(prepared / "prices.csv", calendar)
    benchmark = read_bars(prepared / "benchmark.csv", calendar)
    registry = CompanyRegistry.from_csv(prepared / "companies.csv")
    event_path = Path(projected) / "events.jsonl" if projected else extraction / "events.jsonl"
    projection_manifest = json.loads((Path(projected) / "manifest.json").read_text()) if projected else None
    if projection_manifest and projection_manifest["source_events_sha256"] != sha256_file(extraction / "events.jsonl"):
        raise ValueError("Projected events do not match the supplied extraction run")
    events = list(read_jsonl(event_path))
    relationships = list(read_jsonl(extraction / "relationships.jsonl"))
    policy = EventPolicy(half_life_sessions=config["half_life_sessions"], ttl_sessions=config["ttl_sessions"])
    cutoff = max(s.prediction_at for s in calendar.sessions if s.session_date <= config["sample_end"])
    snapshot = GraphBuilder(calendar, bars, registry, events, relationships, policy).build(cutoff, lookback=len(calendar.sessions))
    if snapshot["contains_synthetic_data"] or embedding_manifest["contains_synthetic_data"]:
        raise ValueError("The real-data study cannot include synthetic inputs")
    study = build_samples(snapshot, bars, benchmark, calendar, config)
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir(); (output / "histories").mkdir()
    write_json(work / "snapshot.json", snapshot)
    write_jsonl(work / "samples.jsonl", study["metadata"])
    np.savez_compressed(work / "samples.npz", features=study["features"], windows=study["windows"], targets=study["targets"])
    write_json(output / "scaler.json", study["scaler"])
    primary_pair = config.get("primary_comparison", ["event_graph_correlation_tft", "event_graph_correlation_gru"])
    plan = {"config": config, "loss_weights": LOSS_WEIGHTS, "variants": list(variants),
            "excluded_models": list(EXCLUDED_MODELS), "primary_comparison": primary_pair,
            "availability_policy": prepared_manifest.get("availability_policy"),
            "tft": {"implementation": "compact TFT following arXiv:1912.09363v3 sections 4.1–4.5",
                    "static_inputs": ["company identity"], "historical_inputs": "18 separate price/market variables plus event and correlation message groups",
                    "known_future_inputs": ["relative horizon index only"], "heads": 4,
                    "objective_adaptation": "Same return, binary-direction and log-volatility heads/loss as GRU; no quantile reproduction claim."},
            "correlation_policy": {"window_sessions": 60, "top_positive_neighbors": 3, "minimum_correlation": 0.2},
            "selection": "All seeds/conditions fixed; checkpoints selected by validation loss; test metrics computed after fitting all runs.",
            "target": "Stock log return minus SPY log return, next session open to fifth following session close",
            "volatility_target": "log RMS of five daily stock-return increments over the same holding interval",
            "graph_statistics": snapshot["stats"], "scaler": study["scaler"],
            "split_counts": {key: int(mask.sum()) for key, mask in study["masks"].items()},
            "input_hashes": {str(p.resolve()): sha256_file(p) for p in [prepared / "manifest.json", prepared / "documents.jsonl",
                             extraction / "manifest.json", extraction / "events.jsonl", extraction / "relationships.jsonl",
                             event_path, embeddings / "manifest.json", embeddings / "vectors.f32", embeddings / "index.jsonl"]},
            "projection": projection_manifest,
            "implementation_hashes": {str(path.relative_to(Path(__file__).parents[2])): sha256_file(path)
                                      for path in Path(__file__).parents[1].rglob("*.py")},
            "contains_synthetic_data": False, "test_evaluated": False}
    write_json(output / "plan.json", plan)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = make_tensor_data(snapshot, vectors, embedding_manifest["dimension"], study, device, config)
    windows = torch.as_tensor(study["windows"], device=device)
    targets = standardized_targets(study, device)
    test = torch.as_tensor(np.flatnonzero(study["masks"]["test"]), device=device)
    validation = torch.as_tensor(np.flatnonzero(study["masks"]["validation"]), device=device)
    replacement = causal_replacement(snapshot["nodes"]["event"])
    control = dict(data)
    replacement_tensor = torch.as_tensor(replacement, device=device)
    control["events"] = data["events"][replacement_tensor]
    control["categories"] = data["categories"][replacement_tensor]
    write_jsonl(output / "historical_content_control.jsonl", [
        {"edge_event_id": event["id"], "content_event_id": snapshot["nodes"]["event"][int(replacement[i])]["id"],
         "edge_known_at": event["known_at"], "content_known_at": snapshot["nodes"]["event"][int(replacement[i])]["known_at"]}
        for i, event in enumerate(snapshot["nodes"]["event"])])
    summaries, validation_predictions = [], {}
    # Preserve the sealed test until all validation selections are finished.
    for variant in variants:
        selected_data = control if variant.startswith("historical_event_control_") else data
        validation_predictions[variant] = []
        for seed in config["seeds"]:
            model, summary = fit_one(variant, seed, selected_data, windows, targets, study, config, output)
            summaries.append(summary)
            validation_predictions[variant].append(restore_predictions(predict(model, selected_data, windows, validation), study["scaler"]))
            del model
    write_json(output / "validation_selection_complete.json", {"runs": summaries, "test_evaluated": False})
    # Only now generate/report held-out test predictions.
    dates = np.array([row["session_date"] for row in study["metadata"]])
    truth = study["targets"][study["masks"]["test"]]
    test_dates = dates[study["masks"]["test"]]
    results, test_predictions, per_seed_predictions = {}, {}, {}
    for variant in variants:
        selected_data = control if variant.startswith("historical_event_control_") else data
        predictions = []
        seed_metrics = []
        for seed in config["seeds"]:
            checkpoint = torch.load(output / "checkpoints" / f"{variant}_seed{seed}.pt", map_location=device, weights_only=True)
            model = make_model(variant, data["prices"].shape[1], embedding_manifest["dimension"], config["hidden_dimension"],
                               len(data["company_ids"]), config["horizon_sessions"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            value = restore_predictions(predict(model, selected_data, windows, test), study["scaler"])
            predictions.append(value)
            seed_metrics.append({"seed": seed, **evaluate(truth, value, test_dates)})
        ensemble = np.mean(predictions, axis=0)
        per_seed_predictions[variant] = np.stack(predictions)
        test_predictions[variant] = ensemble
        validation_ensemble = np.mean(validation_predictions[variant], axis=0)
        results[variant] = {"test": evaluate(truth, ensemble, test_dates), "test_by_seed": seed_metrics,
                            "seed_accuracy_mean": float(np.mean([row["direction_accuracy"] for row in seed_metrics])),
                            "seed_accuracy_std": float(np.std([row["direction_accuracy"] for row in seed_metrics], ddof=1)),
                            "validation": evaluate(study["targets"][study["masks"]["validation"]], validation_ensemble, dates[study["masks"]["validation"]]),
                            "validation_loss_mean": float(np.mean([r["validation_loss"] for r in summaries if r["variant"] == variant]))}
    block, repeats = config["bootstrap_block_sessions"], config["bootstrap_replicates"]
    comparisons = {}
    pairs = [(f"{scenario}_tft", f"{scenario}_gru") for scenario in SCENARIOS]
    pairs += [(f"event_graph_correlation_{arch}", f"price_correlation_{arch}") for arch in ("gru", "tft")]
    pairs += [("event_graph_tft", "same_day_event_tft"), ("event_graph_tft", "historical_event_control_tft")]
    for candidate, baseline in pairs:
        if candidate not in results or baseline not in results: continue
        comparisons[candidate + "__vs__" + baseline] = paired_comparison(truth, test_predictions[candidate], test_predictions[baseline], test_dates, block, repeats)
    active_days = {edge["target"] for edge in snapshot["edges"] if edge["type"] == "ACTIVE_ON"}
    active = np.array([row["day_id"] in active_days for row in study["metadata"]])[study["masks"]["test"]]
    for name, predictions in test_predictions.items():
        if active.any():
            results[name]["test_event_active_days"] = evaluate(truth[active], predictions[active], test_dates[active])
        daily_correct = [float(((predictions[test_dates == date, 1] >= .5) == (truth[test_dates == date, 1] > .5)).mean()) for date in sorted(set(test_dates))]
        results[name]["test_accuracy_interval"] = block_interval(daily_correct, block, repeats)
    np.savez_compressed(output / "test_predictions.npz", targets=truth, dates=test_dates, event_active=active,
                        **{name: values for name, values in test_predictions.items()},
                        **{name + "_seeds": values for name, values in per_seed_predictions.items()})
    write_jsonl(output / "test_samples.jsonl", [row for row in study["metadata"] if row["split"] == "test"])
    primary = comparisons["__vs__".join(primary_pair)]
    summary = {"experiment_id": config["experiment_id"], "status": "completed", "is_forecasting_evaluation": True,
               "is_paper_worthy_improvement_established": False, "contains_synthetic_data": False,
               "config_sha256": digest(config), "device": str(device),
               "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
               "torch_version": str(torch.__version__), "numpy_version": np.__version__,
               "deterministic_algorithms": True, "elapsed_seconds": round(time.monotonic() - started, 3),
               "seeds": config["seeds"], "models": results, "paired_comparisons": comparisons,
               "primary_comparison": primary_pair, "excluded_models": list(EXCLUDED_MODELS),
               "availability_policy": prepared_manifest.get("availability_policy"),
               "is_fresh_holdout": bool(config.get("test_is_fresh_holdout", False)),
               "split_counts": plan["split_counts"], "graph_statistics": snapshot["stats"],
               "source_news_documents": extraction_manifest["summary"]["documents"],
               "raw_extraction_summary": extraction_manifest["summary"], "event_projection": projection_manifest,
               "multi_company_events": sum(len({c["company_id"] for c in e["companies"]}) > 1 for e in events),
               "correlation_edges": int(data["corr_source"].numel()), "test_event_active_samples": int(active.sum()),
               "test_event_active_fraction": float(active.mean()),
               "test_period": {"first": min(test_dates), "last": max(test_dates)},
               "training_runs": summaries,
               "primary_direction_gain_ci_excludes_zero": primary["direction_accuracy_gain"]["ci95"][0] > 0,
               "data_limitations": prepared_manifest["limitations"],
               "method_limitations": [
                   "Single chronological holdout, three optimization seeds; no rolling multi-period replication yet.",
                   "The 2019 benchmark was examined in an earlier experiment; the new architecture comparison is exploratory, not fresh holdout confirmation.",
                   "Extraction has deterministic evidence checks, not a manually labeled semantic-accuracy audit.",
                   "The headline pilot uses separately audited partial projections; ambiguous structured financial fields and unresolved participants are omitted, while original extractions retain review flags.",
                   "Fixed event weights, five-session half-life and twenty-session TTL; no calibrated confidence model.",
                   "One-hop event-to-day aggregation can be expressed as weighted event fusion; it is not evidence for a general GNN advantage.",
                   "No executable trading strategy, transaction costs, or economic-profit claim is evaluated.",
                   "Intervals account for serial and cross-stock dependence via date blocks; multiple comparisons are exploratory.",
               ]}
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": "completed", "output": str(output), "primary_comparison": primary}, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "extraction", "embeddings", "config", "output", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projected", type=Path, help="Optional audited partial-event projection directory")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    run(args.prepared, args.extraction, args.embeddings, config, args.output, args.work, args.device, args.projected)


if __name__ == "__main__":
    main()
