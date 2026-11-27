"""Check real-price target construction and one GPU training epoch, without test metrics."""

import argparse
import json
from pathlib import Path

import torch

from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.graph.builder import GraphBuilder
from stak_graph.graph.market import TradingCalendar, read_bars
from .dataset import build_samples
from .train import fit_one, make_tensor_data, standardized_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Preflight output already exists")
    config = json.loads((args.prepared / "config.json").read_text())
    calendar = TradingCalendar.from_csv(args.prepared / "calendar.csv")
    bars = read_bars(args.prepared / "prices.csv", calendar)
    benchmark = read_bars(args.prepared / "benchmark.csv", calendar)
    registry = CompanyRegistry.from_csv(args.prepared / "companies.csv")
    cutoff = max(s.prediction_at for s in calendar.sessions if s.session_date <= config["sample_end"])
    snapshot = GraphBuilder(calendar, bars, registry, []).build(cutoff, lookback=len(calendar.sessions))
    study = build_samples(snapshot, bars, benchmark, calendar, config)
    args.output.mkdir(); (args.output / "checkpoints").mkdir(); (args.output / "histories").mkdir()
    data = make_tensor_data(snapshot, {}, 768, study, args.device, config)
    windows = torch.as_tensor(study["windows"], device=args.device)
    targets = standardized_targets(study, args.device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config["max_epochs"] = 1
    _, summary = fit_one("price_gru", 7, data, windows, targets, study, config, args.output)
    print(json.dumps({"split_counts": {key: int(value.sum()) for key, value in study["masks"].items()},
                      "scaler_fit_latest": study["scaler"]["fit_latest_date"], "graph_stats": snapshot["stats"],
                      "correlation_edges": int(data["corr_source"].numel()), "preflight": summary,
                      "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
