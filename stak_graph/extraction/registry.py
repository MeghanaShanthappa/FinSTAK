"""Conservative exact alias linking with point-in-time validity."""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import unicodedata


def normalized_name(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


@dataclass(frozen=True)
class Company:
    company_id: str
    name: str
    aliases: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_to: str | None = None

    def __post_init__(self) -> None:
        if not self.company_id or not normalized_name(self.name):
            raise ValueError("Company ID and name must be nonempty")
        for stamp in (self.valid_from, self.valid_to):
            if stamp:
                date.fromisoformat(stamp)
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("Company validity interval is inverted")


class CompanyRegistry:
    def __init__(self, companies: list[Company] | None = None):
        self.companies = companies or []
        self.index: dict[str, list[Company]] = {}
        for company in self.companies:
            for alias in {company.name, *company.aliases}:
                if normalized_name(alias):
                    self.index.setdefault(normalized_name(alias), []).append(company)

    @classmethod
    def from_csv(cls, path: str | Path) -> "CompanyRegistry":
        companies = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"company_id", "name"}.issubset(set(reader.fieldnames or [])):
                raise ValueError("Company CSV must contain company_id and name columns")
            for row in reader:
                if not row.get("company_id") or not row.get("name"):
                    raise ValueError("Company CSV contains a missing company_id or name")
                companies.append(Company(
                    row["company_id"].strip(), row["name"].strip(),
                    tuple(value.strip() for value in (row.get("aliases") or "").split("|") if value.strip()),
                    (row.get("valid_from") or "").strip() or None,
                    (row.get("valid_to") or "").strip() or None,
                ))
        return cls(companies)

    def resolve(self, name: str, as_of: str) -> dict:
        day = as_of[:10]
        matches = {
            company.company_id: company
            for company in self.index.get(normalized_name(name), [])
            if (not company.valid_from or company.valid_from <= day)
            and (not company.valid_to or day <= company.valid_to)
        }
        ids = sorted(matches)
        return {
            "company_id": ids[0] if len(ids) == 1 else None,
            "resolution_status": "resolved" if len(ids) == 1 else "ambiguous" if ids else "unresolved",
            "candidates": ids,
        }
