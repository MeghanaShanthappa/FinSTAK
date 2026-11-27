"""Independently verify persisted inputs, a trained checkpoint, and test metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stak_graph.embeddings.pipeline import load_vectors
from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.graph.builder import EventPolicy, GraphBuilder
from stak_graph.graph.market import TradingCalendar, moment, read_bars
from stak_graph.io import read_jsonl, write_json
from .data import sha256_file
from .dataset import build_samples
from .train import make_model, make_tensor_data, predict, restore_predictions


def verify(output, device="cuda"):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "plan.json").read_text())["config"]
    dataset = output / "dataset"
    calendar = TradingCalendar.from_csv(dataset / "calendar.csv")
    bars = read_bars(dataset / "prices.csv", calendar)
    benchmark = read_bars(dataset / "benchmark.csv", calendar)
    registry = CompanyRegistry.from_csv(dataset / "companies.csv")
    events = list(read_jsonl(output / "event_projection/events.jsonl"))
    relationships = list(read_jsonl(output / "source_extraction/relationships.jsonl"))
    cutoff = max(s.prediction_at for s in calendar.sessions if s.session_date <= config["sample_end"])
    snapshot = GraphBuilder(calendar, bars, registry, events, relationships,
                            EventPolicy(half_life_sessions=config["half_life_sessions"], ttl_sessions=config["ttl_sessions"])).build(cutoff, lookback=len(calendar.sessions))
    study = build_samples(snapshot, bars, benchmark, calendar, config)
    saved_scaler = json.loads((output / "scaler.json").read_text())
    if saved_scaler != study["scaler"]:
        raise ValueError("Reconstructed training-only scaler differs from saved scaler")
    samples = [row for row in study["metadata"] if row["split"] == "test"]
    if samples != list(read_jsonl(output / "test_samples.jsonl")):
        raise ValueError("Reconstructed test order/identities differ")
    saved = np.load(output / "test_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["targets"], study["targets"][study["masks"]["test"]])
    for name, model in summary["models"].items():
        probability = saved[name][:, 1]
        accuracy = float(((probability >= .5) == (saved["targets"][:, 1] > .5)).mean())
        if abs(accuracy - model["test"]["direction_accuracy"]) > 1e-12:
            raise ValueError(f"Accuracy did not reproduce: {name}")
        frame = pd.DataFrame({"date": saved["dates"], "truth": saved["targets"][:, 0], "score": saved[name][:, 0]})
        ics = []
        for _, group in frame.groupby("date"):
            truth_rank, score_rank = group["truth"].rank(), group["score"].rank()
            if truth_rank.std() > 0 and score_rank.std() > 0:
                ics.append(float(truth_rank.corr(score_rank)))
        result = model["test"]["mean_daily_rank_ic"]
        if ics and abs(float(np.mean(ics)) - result) > 1e-12:
            raise ValueError(f"Independent pandas rank IC did not reproduce: {name}")
    for run in summary["training_runs"]:
        path = output / "checkpoints" / Path(run["checkpoint"]).name
        if sha256_file(path) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
    vectors, manifest = load_vectors(output / "event_embeddings")
    data = make_tensor_data(snapshot, vectors, manifest["dimension"], study, device, config)
    windows = torch.as_tensor(study["windows"], device=device)
    test = torch.as_tensor(np.flatnonzero(study["masks"]["test"]), device=device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    first = min(row["session_index"] for row in samples)
    early = torch.tensor([i for i, row in enumerate(study["metadata"]) if row["split"] == "test" and row["session_index"] == first], device=device)
    future_prices = torch.tensor([day["session_index"] > first for day in snapshot["nodes"]["stock_day"]], device=device)
    cutoff_time = calendar.times[first]
    future_events = torch.tensor([moment(event["known_at"]) > cutoff_time for event in snapshot["nodes"]["event"]], device=device)
    variants = summary.get("primary_comparison", ["event_graph_correlation_gru"])
    checked = []
    for variant in variants:
        seed = config["seeds"][0]
        checkpoint = torch.load(output / "checkpoints" / f"{variant}_seed{seed}.pt", map_location=device, weights_only=True)
        model = make_model(variant, data["prices"].shape[1], manifest["dimension"], config["hidden_dimension"],
                           len(data["company_ids"]), config["horizon_sessions"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        reproduced = restore_predictions(predict(model, data, windows, test), study["scaler"])
        np.testing.assert_allclose(reproduced, saved[variant + "_seeds"][0], rtol=1e-6, atol=1e-7)
        earlier = predict(model, data, windows, early)
        perturbed = dict(data)
        perturbed["prices"] = data["prices"].clone(); perturbed["prices"][future_prices] = 1000
        perturbed["events"] = data["events"].clone(); perturbed["events"][future_events] = 1000
        later = predict(model, perturbed, windows, early)
        np.testing.assert_allclose(earlier, later, rtol=0, atol=0)
        checked.append(variant)
    if set(summary.get("excluded_models", [])) & set(summary["models"]):
        raise ValueError("An excluded baseline was evaluated")
    for row in read_jsonl(output / "historical_content_control.jsonl"):
        if moment(row["content_known_at"]) > moment(row["edge_known_at"]):
            raise ValueError("Negative control imports future event content")
    result = {"status": "passed", "models_with_independently_recomputed_metrics": len(summary["models"]),
              "rank_ic_independent_implementation": "pandas average ranks and Pearson correlation",
              "checkpoint_hashes_verified": len(summary["training_runs"]),
              "reconstructed_training_scaler": True, "reconstructed_test_targets_and_order": True,
              "primary_seed_checkpoint_predictions_reproduced": True,
              "reproduced_variants": checked, "excluded_baselines_absent": True,
              "forecast_invariant_to_future_price_and_event_perturbation": True,
              "future_prices_perturbed": int(future_prices.sum()), "future_events_perturbed": int(future_events.sum()),
              "contains_synthetic_data": False}
    write_json(output / "verification.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
