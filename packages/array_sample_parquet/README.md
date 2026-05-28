# array-sample-parquet

`array-sample-parquet`은 array trace를 sample-major Parquet dataset으로 쓰고 읽는 Python 패키지입니다. 기존 custom binary array shard보다 직접 열어보기 쉽고, viewer/debugging 또는 sample 중심 조회에 맞춰져 있습니다.

## Public API

- `ArraySampleParquetDatasetBuilder.open_session(...)`: resumable build session을 엽니다.
- `builder.status()`: `next_expected_sample_id`를 포함한 재개 위치를 확인합니다.
- `builder.sample(sample_id=...)` / `builder.sample(sample_key=...)`: sample 경계를 엽니다.
- `sample.add_trace(...)`: 현재 sample 안에 feature trace 하나를 추가합니다.
- `builder.finish()`: 남은 part를 commit하고 `array_sample_parquet_manifest.json`을 생성합니다.
- `open_array_sample_parquet(manifest_path)`: reader를 엽니다.
- `reader.get_traces(...)`: id/key 조건으로 trace 객체를 읽습니다.
- `reader.get_traces_json(layout="nested"|"flat")`: API 응답과 같은 JSON-compatible layout으로 반환합니다.

## Format Summary

artifact 구조는 다음과 같습니다.

```text
out_dir/
  array_sample_parquet_manifest.json
  state.json
  parts.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_parts/
    part_000000.parquet
  categorical_dictionaries/
    ch_step.json
```

`sample_parts/*.parquet`의 row 하나는 `(sample_id, feature_id)` trace 하나입니다. point column은 Parquet `list<typed value>`로 저장합니다.

```text
sample_id  int64
feature_id int32
trace_len  int32
time       list<float64>
value      list<float64>
ch_step    list<uint32>
```

missing trace는 row 부재로 표현하고, empty trace는 `trace_len=0`인 row로 표현합니다. point-level null은 지원하지 않습니다.

## Build Behavior

builder는 전체 part를 메모리에 쌓지 않습니다. `add_trace(...)`가 호출되면 trace row를 즉시 `.parquet.tmp` writer에 전달하고, 작은 PyArrow row batch만 메모리에 유지합니다.

part는 sample 경계에서만 commit됩니다. flush 기준은 sample 개수가 아니라 `target_part_bytes`입니다. `max_part_rows`와 `max_part_samples`는 안전장치입니다.

중간에 프로세스가 종료되면 `parts.jsonl`에 기록된 part만 committed로 인정합니다. resume 시 `.parquet.tmp`는 삭제되고, 사용자는 `builder.status().next_expected_sample_id`부터 다시 데이터를 넣으면 됩니다.

## Example

```python
from array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
    open_array_sample_parquet,
)

schema = [
    PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
    PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
    PointColumnSpec("ch_step", StorageType.UINT32, LogicalType.CATEGORICAL),
]

options = ArraySampleParquetBuildOptions(
    target_part_bytes=128 * 1024 * 1024,
    max_part_rows=100_000,
    max_part_samples=0,
    compression="zstd",
)

with ArraySampleParquetDatasetBuilder.open_session(
    "data/array_sample_parquet",
    "data/sample_meta.parquet",
    schema,
    feature_meta_path="data/feature_meta.parquet",
    options=options,
) as builder:
    start = builder.status().next_expected_sample_id
    for sample_id in range(start, n_samples):
        with builder.sample(sample_id=sample_id) as sample:
            sample.add_trace(
                feature_key="feature_a",
                columns={
                    "time": [0.0, 1.0],
                    "value": [10.0, 11.0],
                    "ch_step": ["idle", "run"],
                },
            )
    manifest_path = builder.finish()

reader = open_array_sample_parquet(manifest_path)
result = reader.get_traces_json(
    sample_keys=["sample_000001"],
    feature_keys=["feature_a"],
    decode_categorical=True,
    include_missing=True,
    layout="nested",
)
```

## When To Use

이 패키지는 sample 중심 viewer/debugging에 적합합니다. feature 하나를 모든 sample에서 반복 조회하는 serving workload에는 기존 `array-binary-shard`가 더 적합합니다.

자세한 포맷, 구현 방식, API 서버 요청/응답은 `docs/array_sample_parquet_format_v1.md`를 참고하십시오.
