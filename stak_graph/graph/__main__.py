import argparse
from pathlib import Path
import sys

from stak_graph.extraction.registry import CompanyRegistry
from stak_graph.extraction.schema import canonical_json, digest
from stak_graph.io import read_jsonl, write_json
from .builder import EventPolicy, GraphBuilder
from .market import TradingCalendar, read_bars


def main():
    parser = argparse.ArgumentParser(description="Build a point-in-time STAK-Graph snapshot")
    for name in ("events", "prices", "calendar", "registry", "cutoff", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--relationships")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--half-life-sessions", type=float, default=5)
    parser.add_argument("--ttl-sessions", type=int, default=20)
    parser.add_argument("--propagation-weight", type=float, default=0.5)
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists():
            raise ValueError("Graph output exists; choose a new directory")
        calendar = TradingCalendar.from_csv(args.calendar)
        policy = EventPolicy(args.half_life_sessions, args.ttl_sessions, 1.0, args.propagation_weight, not args.direct_only)
        snapshot = GraphBuilder(calendar, read_bars(args.prices, calendar), CompanyRegistry.from_csv(args.registry),
                                read_jsonl(args.events), read_jsonl(args.relationships) if args.relationships else [], policy).build(args.cutoff, args.lookback)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "snapshot.json", snapshot)
        inputs = {key: digest(Path(getattr(args, key)).read_text()) for key in ("events", "prices", "calendar", "registry")}
        if args.relationships:
            inputs["relationships"] = digest(Path(args.relationships).read_text())
        write_json(output / "manifest.json", {"input_hashes": inputs, "snapshot_hash": digest(snapshot), "stats": snapshot["stats"],
                                              "contains_synthetic_data": snapshot["contains_synthetic_data"], "is_forecasting_evaluation": False})
        print(canonical_json(snapshot["stats"]))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Graph error: {exc}", file=sys.stderr)
        return 2


raise SystemExit(main())
