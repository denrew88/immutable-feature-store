"""JSON helper functions for array_sample_parquet build logs."""

from __future__ import annotations

import json
import os
import time
import uuid

from .file_lock import FilePathLock


def write_json_atomic(path: str, payload: dict):
    """Write JSON through a unique temporary file so readers never see a partial file."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    lock = FilePathLock(path + ".lock")
    lock.acquire()
    try:
        # final JSON을 직접 덮어쓰지 않고 UUID tmp에 먼저 씁니다.
        # reader는 final path만 보기 때문에 partial JSON을 관측하지 않습니다.
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        last_error = None
        for attempt in range(8):
            try:
                # Windows에서 백신/IDE가 final JSON handle을 잠깐 잡는 경우를 방어합니다.
                os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_error = exc
                if attempt == 7:
                    break
                time.sleep(0.025 * float(attempt + 1))
        raise last_error
    finally:
        lock.release()
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


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
