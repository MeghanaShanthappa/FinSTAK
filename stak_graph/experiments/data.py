"""Download pinned FNSPID artifacts and prepare a label-independent pilot.

Historical news availability is not observed in this release. We deliberately
delay every headline until two calendar days after its recorded source date.
This is a conservative assumption, not a verified point-in-time news feed.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

from stak_graph.io import write_json, write_jsonl
from stak_graph.extraction.schema import digest
from stak_graph.extraction.sources import Document

ARTIFACTS = {
    "All_external.csv": ("Stock_news/All_external.csv", 5731397037,
                         "5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3"),
    "full_history.zip": ("Stock_price/full_history.zip", 589525596,
                         "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91"),
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def obtain_artifact(root, filename, revision):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    relative, size, checksum = ARTIFACTS[filename]
    target = root / filename
    if not target.exists():
        url = f"https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/{revision}/{relative}?download=true"
        partial = target.with_suffix(".part")
        with urlopen(url, timeout=120) as response, partial.open("xb") as output:
            if response.status != 200:
                raise ValueError("Dataset download must contain the complete artifact")
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
        partial.rename(target)
    if target.stat().st_size != size or sha256_file(target) != checksum:
        raise ValueError(f"Dataset artifact checksum mismatch: {target}")
    return target


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def delayed_availability(source_timestamp, delay_days):
    """Ignore uncertain time-of-day; do not backfill at source-date midnight."""
    published = datetime.strptime(source_timestamp.strip(), "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    known = published.replace(hour=0, minute=0, second=0) + timedelta(days=delay_days)
    if known < published or delay_days < 1:
        raise ValueError("Conservative availability must follow publication")
    return published.isoformat().replace("+00:00", "Z"), known.isoformat().replace("+00:00", "Z")


def same_day_availability(published_at):
    """Preserve reported clocks; date-only midnight placeholders become EOD.

    EOD is 23:59:59 UTC on the recorded source date. A midnight placeholder
    is never represented as verified intraday publication. Session forecast
    cutoffs must also be EOD when this policy is used.
    """
    from stak_graph.extraction.sources import utc_timestamp
    published = datetime.fromisoformat(utc_timestamp(published_at).replace("Z", "+00:00"))
    date_only = published.hour == published.minute == published.second == published.microsecond == 0
    known = published.replace(hour=23, minute=59, second=59) if date_only else published
    return known.isoformat().replace("+00:00", "Z"), "date_only_assumed_end_of_day" if date_only else "reported_time_unverified"


def headline_mentions(text, company):
    return any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
               for alias in [company["name"], *company["aliases"]])


def extractor_documents(documents):
    inputs = [{key: value for key, value in row.items() if key not in ("source_row", "source_stock_symbols")}
              for row in documents]
    for row in inputs:
        Document.from_dict(row)
    return inputs


def select_headlines(source_path, config):
    from .events import CATEGORY_PATTERNS, category_surface_supported
    companies = {row["company_id"]: row for row in config["companies"]}
    candidates = {}
    counts = {"rows_scanned": 0, "target_stock_rows": 0, "in_period": 0,
              "eligible_named_headlines": 0, "invalid_source_rows": 0}
    csv.field_size_limit(64 * 1024 * 1024)
    with Path(source_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"Date", "Article_title", "Stock_symbol", "Url"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected FNSPID news schema")
        for row in reader:
            counts["rows_scanned"] += 1
            ticker = row["Stock_symbol"]
            if ticker not in companies:
                continue
            counts["target_stock_rows"] += 1
            stamp = row["Date"] or ""
            if not config["sample_start"] <= stamp[:10] <= config["sample_end"]:
                continue
            counts["in_period"] += 1
            title, url = (row["Article_title"] or "").strip(), (row["Url"] or "").strip()
            if not title or not url.startswith(("https://", "http://")):
                counts["invalid_source_rows"] += 1
                continue
            if not headline_mentions(title, companies[ticker]):
                continue
            try:
                if config.get("news_availability") == "same_day":
                    published = datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
                    known, _ = same_day_availability(published)
                else:
                    published, known = delayed_availability(stamp, config["availability_delay_calendar_days"])
            except ValueError:
                counts["invalid_source_rows"] += 1
                continue
            counts["eligible_named_headlines"] += 1
            identifier = digest([url, title])
            if identifier not in candidates:
                candidates[identifier] = {
                    "document_id": "fnspid_" + identifier[:24], "text": title, "title": title,
                    "source_url": url, "published_at": published, "known_at": known,
                    "source_type": "fnspid_original_headline", "is_synthetic": False,
                    "source_row": reader.line_num, "source_stock_symbols": [ticker],
                }
            else:
                # A duplicate with a later date may be an update: retain the
                # later availability rather than leaking a revised headline.
                item = candidates[identifier]
                item["published_at"] = max(item["published_at"], published)
                item["known_at"] = max(item["known_at"], known)
                item["source_stock_symbols"] = sorted(set(item["source_stock_symbols"] + [ticker]))
    buckets = {}
    for identifier, item in candidates.items():
        rank = digest([config["headline_sampling_salt"], identifier])
        for ticker in item["source_stock_symbols"]:
            bucket = (ticker, item["published_at"][:7])
            buckets.setdefault(bucket, []).append((rank, item["document_id"]))
    selected = set()
    for values in buckets.values():
        selected.update(identifier for _, identifier in sorted(values)[:config["headlines_per_company_month"]])
    event_buckets = {}
    if config.get("event_focused_headlines_per_company_month", 0):
        by_id = {row["document_id"]: row for row in candidates.values()}
        for bucket, values in buckets.items():
            qualifying = [(rank, identifier) for rank, identifier in values
                          if any(category_surface_supported(category, by_id[identifier]["text"]) for category in CATEGORY_PATTERNS)]
            if qualifying:
                event_buckets[bucket] = qualifying
                selected.update(identifier for _, identifier in sorted(qualifying)[:config["event_focused_headlines_per_company_month"]])
    documents = sorted((item for item in candidates.values() if item["document_id"] in selected),
                       key=lambda row: (row["known_at"], row["document_id"]))
    counts.update({"unique_candidates": len(candidates), "company_month_buckets": len(buckets),
                   "selected_documents": len(documents), "event_focused_buckets": len(event_buckets),
                   "selection_uses_price_labels": False})
    return documents, counts


def prepare(raw_root, output, config):
    import exchange_calendars as xcals
    import pandas as pd

    output = Path(output)
    if output.exists():
        raise ValueError("Prepared-data output already exists")
    news = obtain_artifact(raw_root, "All_external.csv", config["dataset_revision"])
    archive = obtain_artifact(raw_root, "full_history.zip", config["dataset_revision"])
    # Extend beyond the final prediction date so the last held-out labels can mature.
    calendar_end = (datetime.fromisoformat(config["sample_end"]) + timedelta(days=20)).date().isoformat()
    calendar = xcals.get_calendar("XNYS", start=config["calendar_start"], end=calendar_end)
    schedule = calendar.schedule.loc[config["calendar_start"]:calendar_end]
    sessions = []
    for stamp, row in schedule.iterrows():
        close = row["close"]
        sessions.append({"session_date": stamp.date().isoformat(), "open_at": row["open"].isoformat(),
                         "close_at": close.isoformat(),
                         "prediction_at": stamp.date().isoformat() + "T23:59:59Z" if config.get("news_availability") == "same_day"
                         else (close + pd.Timedelta(minutes=config["bar_delay_minutes"])).isoformat(),
                         "is_synthetic": "false"})
    session_by_date = {row["session_date"]: row for row in sessions}
    prices, benchmark = [], []
    source_counts = {}
    with zipfile.ZipFile(archive) as source:
        for ticker in [row["company_id"] for row in config["companies"]] + ["SPY"]:
            with source.open(f"full_history/{ticker}.csv") as binary:
                reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
                selected = []
                for row in reader:
                    if config["calendar_start"] <= row["date"] <= calendar_end:
                        if row["date"] not in session_by_date:
                            raise ValueError("Source price date is absent from the XNYS calendar")
                        factor = float(row["adj close"]) / float(row["close"])
                        selected.append({
                            "company_id": ticker, "session_date": row["date"],
                            **{key: float(row[key]) * factor for key in ("open", "high", "low", "close")},
                            "volume": float(row["volume"]),
                            "available_at": (pd.Timestamp(session_by_date[row["date"]]["close_at"]) + pd.Timedelta(minutes=config["bar_delay_minutes"])).isoformat(), "is_synthetic": "false",
                        })
            selected.sort(key=lambda row: row["session_date"])
            if [row["session_date"] for row in selected] != list(session_by_date):
                raise ValueError(f"Incomplete pilot price coverage for {ticker}; revise universe explicitly")
            source_counts[ticker] = len(selected)
            (benchmark if ticker == "SPY" else prices).extend(selected)
    documents, counts = select_headlines(news, config)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "calendar.csv", list(sessions[0]), sessions)
    write_csv(output / "prices.csv", list(prices[0]), prices)
    write_csv(output / "benchmark.csv", list(benchmark[0]), benchmark)
    registry = [{"company_id": row["company_id"], "name": row["name"],
                 "aliases": "|".join(row["aliases"]), "valid_from": config["calendar_start"], "valid_to": calendar_end}
                for row in config["companies"]]
    write_csv(output / "companies.csv", list(registry[0]), registry)
    write_jsonl(output / "news_provenance.jsonl", documents)
    extractor_inputs = extractor_documents(documents)
    write_jsonl(output / "documents.jsonl", extractor_inputs)
    write_json(output / "config.json", config)
    manifest = {
        "dataset": "Zihan1004/FNSPID", "revision": config["dataset_revision"],
        "source": "https://huggingface.co/datasets/Zihan1004/FNSPID",
        "source_artifacts": {key: {"bytes": value[1], "sha256": value[2]} for key, value in ARTIFACTS.items()},
        "calendar": {"name": "XNYS", "package": "exchange_calendars", "version": xcals.__version__,
                     "sessions": len(sessions), "early_closes": int(sum((pd.Timestamp(row["close_at"]).tz_convert("America/New_York").hour < 16) for row in sessions))},
        "price_rows_by_ticker": source_counts, "news_selection": counts,
        "config_sha256": digest(config),
        "files": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()},
        "contains_synthetic_data": False,
        "limitations": [
            "Fixed, manually chosen large-company universe; survivorship/generalization are not addressed.",
            ("Date-only headlines are assumed available at 23:59:59 UTC on the recorded date, with EOD forecasts; historical arrival times are unverified."
             if config.get("news_availability") == "same_day" else "News known_at is an assumed two-calendar-day delay from recorded source date, not a verified historical arrival time."),
            "Source timestamps may be rounded or shifted; headline updates are not versioned in the original release.",
            "Historical OHLC is retrospectively adjusted using adjusted_close/close; raw volume is retained.",
            "Frozen pretrained language models may have seen historical articles and future facts during pretraining.",
            "Headline-only, stratified monthly sampling is a sparse pilot, not exhaustive event coverage; event vocabulary is used when enabled, without return labels.",
        ],
        "license_at_pinned_dataset_card": "CC BY-NC 4.0; see original dataset card and upstream news rights",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.raw_root, args.output, json.loads(args.config.read_text()))
    print(json.dumps({"prices": result["price_rows_by_ticker"], "news": result["news_selection"], "calendar": result["calendar"]}, indent=2))


if __name__ == "__main__":
    main()
