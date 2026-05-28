"""Categorical dictionary helpers for the sample-major Parquet format."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def write_json_atomic(path: str, payload: dict):
    """JSON 파일을 `.tmp`에 먼저 쓴 뒤 rename해서 부분 파일을 남기지 않는다."""

    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def append_jsonl(path: str, payload: dict):
    """commit log에 JSON object 한 줄을 append한다.

    parts.jsonl은 이미 commit된 part만 기록한다. 재시작 시에는 이 로그만 믿고
    orphan `.parquet` 파일은 무시하므로 중간 실패에도 resume 경계가 명확하다.
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def load_jsonl(path: str) -> list[dict]:
    """JSONL 파일을 앞에서부터 읽고, 깨진 tail line은 commit되지 않은 것으로 본다."""

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


@dataclass
class CategoricalRegistry:
    """문자 label을 uint32 code로 바꾸는 mutable dictionary.

    code 0은 missing/unknown 예약값으로 남겨두고, 실제 label은 1부터 배정한다.
    같은 registry 상태가 part commit log에 같이 기록되므로 resume 후에도 이미
    저장된 part의 code 의미가 바뀌지 않는다.
    """

    label_to_code: dict[str, int]
    code_to_label: list[str]

    @classmethod
    def create(cls) -> "CategoricalRegistry":
        return cls(label_to_code={}, code_to_label=[])

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> "CategoricalRegistry":
        registry = cls.create()
        for label in labels:
            text = str(label)
            if text not in registry.label_to_code:
                registry.label_to_code[text] = len(registry.code_to_label) + 1
                registry.code_to_label.append(text)
        return registry

    def encode(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=object).reshape(-1)
        out = np.zeros(int(arr.size), dtype=np.uint32)
        for idx, value in enumerate(arr.tolist()):
            if value is None:
                out[idx] = np.uint32(0)
                continue
            label = str(value)
            code = self.label_to_code.get(label)
            if code is None:
                code = len(self.code_to_label) + 1
                self.label_to_code[label] = int(code)
                self.code_to_label.append(label)
            out[idx] = np.uint32(code)
        return out

    def labels_json(self) -> list[str]:
        return [str(label) for label in self.code_to_label]

    def to_json_dict(self, column_name: str) -> dict:
        return {
            "column": str(column_name),
            "items": [
                {"code": int(idx + 1), "label": str(label)}
                for idx, label in enumerate(self.code_to_label)
            ],
        }


def load_categorical_dictionary(path: str) -> dict[int, str]:
    """sidecar dictionary JSON을 `{code: label}` mapping으로 읽는다."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, str] = {}
    if isinstance(data.get("items"), list):
        for item in data["items"]:
            out[int(item["code"])] = str(item["label"]) if item.get("label") is not None else None
        return out
    labels = data.get("labels")
    if isinstance(labels, dict):
        return {int(code): str(label) if label is not None else None for code, label in labels.items()}
    raise ValueError(f"unsupported categorical dictionary format: {path}")
