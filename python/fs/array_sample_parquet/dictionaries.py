"""JSON helper functions for array_sample_parquet build logs."""

from __future__ import annotations

import json
import os


def write_json_atomic(path: str, payload: dict):
    """Write JSON through a temporary file so readers never see a partial file."""

    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def append_jsonl(path: str, payload: dict):
    """Append one committed JSON object to a JSONL log."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def load_jsonl(path: str) -> list[dict]:
    """Read committed JSONL rows and ignore a broken tail line."""

    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out
