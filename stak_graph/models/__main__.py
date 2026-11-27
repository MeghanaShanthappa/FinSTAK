"""A forward/backward architecture check, explicitly not a forecast evaluation."""

import argparse
from pathlib import Path

import torch

from stak_graph.embeddings.pipeline import load_vectors
from stak_graph.extraction.schema import canonical_json, strict_json
from stak_graph.io import write_json
from .event_gru import EventGraphGRU, tensorize


def main():
    parser = argparse.ArgumentParser(description="STAK-Graph model architecture smoke test")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists")
    torch.manual_seed(7)
    snapshot = strict_json(Path(args.snapshot).read_text())
    vectors, manifest = load_vectors(args.embeddings)
    data = tensorize(snapshot, vectors, manifest["dimension"], args.device)
    model = EventGraphGRU(len(snapshot["feature_names"]), manifest["dimension"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction = model(data)
    loss = sum(value.square().mean() for value in prediction.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(bool(g.abs().sum() > 0) for g in gradients)
    if not finite or nonzero == 0:
        raise ValueError("Forward/backward pass did not produce finite nonzero gradients")
    optimizer.step()
    result = {"model": "EventGraphGRU", "device": args.device, "stock_days": len(data["day_ids"]),
              "companies": len(data["sequences"]), "event_dimension": manifest["dimension"],
              "active_edges": len(data["edge_weight"]), "finite_forward_backward": finite,
              "nonzero_gradient_tensors": nonzero, "optimizer_steps": 1,
              "parameters": sum(p.numel() for p in model.parameters()),
              "contains_synthetic_data": snapshot["contains_synthetic_data"],
              "embedding_is_simulated": manifest["encoder"]["simulated"],
              "is_trained_forecaster": False, "is_forecasting_evaluation": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model_smoke.json", result)
    print(canonical_json(result))


main()
