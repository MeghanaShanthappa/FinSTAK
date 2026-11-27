"""Finish a running extraction job, embed events, fit baselines, and save results.

Run in the same Python environment as the model stages. No unrelated process
is stopped or modified. Existing stage outputs are never overwritten.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "extraction", "projected", "embeddings", "config", "output", "work", "state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--model-cache", type=Path, default=Path("/tmp/stak-graph-models"))
    parser.add_argument("--timeout-hours", type=float, default=4)
    args = parser.parse_args(argv)
    start = time.monotonic()

    def state(stage, **kwargs):
        value = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": round(time.monotonic() - start, 3), **kwargs}
        partial = args.state.with_suffix(".tmp")
        partial.write_text(json.dumps(value, indent=2) + "\n")
        partial.replace(args.state)
        print(json.dumps(value), flush=True)

    def run_stage(stage, command):
        state(stage, command=command)
        subprocess.run(command, check=True)

    try:
        if args.output.exists():
            raise ValueError("Forecast output already exists; choose a fresh directory")
        state("waiting_for_extraction")
        while not (args.extraction / "manifest.json").exists():
            if time.monotonic() - start > args.timeout_hours * 3600:
                raise TimeoutError("Extraction did not finish before the configured timeout")
            if args.wait_pid:
                os.kill(args.wait_pid, 0)
            time.sleep(15)
        py = sys.executable
        run_stage("projecting_supported_event_fields", [py, "-m", "stak_graph.experiments.events",
                  "--extraction", str(args.extraction), "--output", str(args.projected)])
        run_stage("embedding_events_on_gpu", [py, "-m", "stak_graph.embeddings", "--events", str(args.projected / "events.jsonl"),
                  "--output", str(args.embeddings), "--model", "google/embeddinggemma-2",
                  "--revision", "f9972ac22e696a105c482202646979e33e4b5a1e", "--device", "cuda",
                  "--model-cache", str(args.model_cache), "--local-files-only", "--cache", "/tmp/stak-graph-real-vectors.sqlite3"])
        run_stage("training_and_evaluating", [py, "-m", "stak_graph.experiments.train", "--prepared", str(args.prepared),
                  "--extraction", str(args.extraction), "--projected", str(args.projected), "--embeddings", str(args.embeddings),
                  "--config", str(args.config), "--output", str(args.output), "--work", str(args.work), "--device", "cuda"])
        state("saving_reproducible_inputs")
        sources = {"dataset": args.prepared, "source_extraction": args.extraction,
                   "event_projection": args.projected, "event_embeddings": args.embeddings}
        required = sum(path.stat().st_size for source in sources.values() for path in source.rglob("*") if path.is_file())
        if shutil.disk_usage(args.output).free < required + 5 * 1024 * 1024:
            raise OSError("Insufficient home storage to preserve the small reproducibility inputs")
        for name, source in sources.items():
            shutil.copytree(source, args.output / name)
        runtime = {"python": sys.version, "packages": {}}
        for name in ("numpy", "pandas", "torch", "transformers", "xgrammar", "exchange_calendars"):
            try: runtime["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        run_stage("verifying_saved_checkpoints_and_metrics", [py, "-m", "stak_graph.experiments.verify", "--output", str(args.output)])
        run_stage("generating_results_report", [py, "-m", "stak_graph.experiments.report", "--output", str(args.output)])
        state("completed", results=str((args.output / "summary.json").resolve()), report=str((args.output / "results.html").resolve()))
    except BaseException as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
