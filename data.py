"""Data helpers for Homework 4 preference JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping


PreferenceRecord = Dict[str, str]


def read_jsonl(path: str | Path) -> List[PreferenceRecord]:
    """Read a JSONL file of {"prompt", "chosen", "rejected"} rows."""
    path = Path(path)
    records: List[PreferenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("prompt", "chosen", "rejected"):
                if field not in record:
                    raise ValueError(f"Missing field '{field}' on line {line_number} of {path}.")
                if not isinstance(record[field], str):
                    raise TypeError(
                        f"Field '{field}' must be a string on line {line_number} of {path}."
                    )
            records.append(
                {
                    "prompt": record["prompt"],
                    "chosen": record["chosen"],
                    "rejected": record["rejected"],
                }
            )
    return records


def write_jsonl(records: Iterable[Mapping[str, str]], path: str | Path) -> None:
    """Write JSONL preference data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
