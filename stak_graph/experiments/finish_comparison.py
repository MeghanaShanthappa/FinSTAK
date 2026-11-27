"""Train, verify and publish a prepared same-day TFT/GRU comparison."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

from .data import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args(argv)
    root, publish = args.work_root.resolve(), args.publish.resolve()
    output = root / "results"
    started = time.monotonic()

    def state(stage, **kwargs):
        value = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": round(time.monotonic() - started, 3), **kwargs}
        temporary = root / "comparison_state.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(root / "comparison_state.json")
        print(json.dumps(value), flush=True)

    def stage(name, command):
        state(name)
        subprocess.run([sys.executable, "-m", *command], check=True)

    try:
        if publish.exists() or output.exists():
            raise ValueError("Comparison output already exists")
        stage("training_and_evaluating", ["stak_graph.experiments.train", "--prepared", str(root / "prepared"),
              "--extraction", str(root / "extraction"), "--projected", str(root / "projected"),
              "--embeddings", str(root / "embeddings"), "--config", str(args.config), "--output", str(output),
              "--work", str(root / "study"), "--device", "cuda"])
        state("saving_processed_inputs")
        for name, source in {"dataset": "prepared", "source_extraction": "extraction", "event_projection": "projected", "event_embeddings": "embeddings"}.items():
            shutil.copytree(root / source, output / name)
        runtime = {"python": sys.version, "packages": {}}
        for name in ("numpy", "pandas", "torch", "transformers", "xgrammar"):
            runtime["packages"][name] = importlib.metadata.version(name)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        stage("verifying_both_backbones", ["stak_graph.experiments.verify", "--output", str(output), "--device", "cuda"])
        stage("creating_comparison_report", ["stak_graph.experiments.report", "--output", str(output)])
        state("archiving_selected_models")
        archive = root / "full_artifacts.tar.gz"
        with tarfile.open(archive, "x:gz") as target:
            target.add(output, arcname=publish.name)
        compact_names = ("results.html", "summary.json", "plan.json", "verification.json", "artifact_hashes.json", "test_samples.jsonl", "scaler.json", "runtime.json")
        compact_size = sum((output / name).stat().st_size for name in compact_names)
        publish.parent.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(publish.parent).free
        if available < compact_size + 3 * 1024 * 1024:
            raise OSError("Insufficient storage even for the compact results report")
        publish.mkdir()
        for name in compact_names: shutil.copyfile(output / name, publish / name)
        persistent_archive = available >= compact_size + archive.stat().st_size + 3 * 1024 * 1024
        archive_target = publish / archive.name if persistent_archive else archive
        if persistent_archive: shutil.copyfile(archive, archive_target)
        publication = {"full_artifacts_archive": str(archive_target), "archive_sha256": sha256_file(archive_target),
                       "archive_bytes": archive_target.stat().st_size, "archive_saved_in_project": persistent_archive,
                       "uncompressed_temporary_run": str(output),
                       "artifact_hashes_scope": "All files inside the full experiment archive; the publication folder contains compact copies.",
                       "reconstruction": f"Extract the archive and run python -m stak_graph.experiments.verify --output PATH/{publish.name}"}
        (publish / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")
        archive_link = "full_artifacts.tar.gz" if persistent_archive else str(archive)
        with (publish / "results.html").open("a") as report:
            report.write(f'<p style="font-family:system-ui;padding:22px"><a href="{archive_link}">Full saved models, predictions, data and histories</a> · <a href="publication.json">Archive details and reconstruction</a></p>')
        state("completed", report=str(publish / "results.html"), summary=str(publish / "summary.json"), **publication)
    except BaseException as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
