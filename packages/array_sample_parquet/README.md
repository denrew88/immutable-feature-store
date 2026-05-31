# array-sample-parquet

`array-sample-parquet`는 array trace를 sample-major long Parquet dataset으로 쓰고 읽는 Python 패키지입니다. custom binary array shard보다 직접 열어보기 쉽고, viewer/debugging 또는 sample 중심 조회에 맞춰져 있습니다.

## Public API

- `ArraySampleParquetDatasetBuilder.open_session(...)`: resumable build session을 엽니다.
- `builder.status()`: 완료/미완료 sample 현황과 manifest 생성 여부를 확인합니다.
- `builder.pending_sample_ids()`: 아직 완료되지 않은 sample id 목록을 반환합니다.
- `builder.sample(sample_id=...)` / `builder.sample(sample_key=...)`: sample context를 엽니다.
- `sample.add_trace(...)`: 현재 sample 안에 feature trace 하나를 추가합니다.
- `builder.finish()`: 모든 sample이 완료되었는지 확인하고 final part를 생성합니다.
- `builder.compact()`: raw sample 파일들을 final part parquet로 묶고 manifest를 생성합니다.
- `open_array_sample_parquet(manifest_path)`: reader를 엽니다.
- `reader.get_traces(...)`: id/key 조건으로 trace 객체를 읽습니다.
- `reader.get_traces_json(layout="nested"|"flat")`: API 응답과 같은 JSON-compatible layout으로 반환합니다.

## Format Summary

최종 physical layout은 long format입니다.

- `sample_parts/*.parquet`: `sample_id`, `feature_id`, `point_idx`, point column을 가진 point rows입니다.
- `trace_index_parts/*.parquet`: `sample_id`, `feature_id`, `trace_len`을 가진 present trace index입니다.
- empty trace는 trace index에 `trace_len=0`으로 있고 point row는 없습니다.
- missing trace는 trace index row 자체가 없습니다.
- categorical point column은 sidecar dictionary 없이 string primitive column으로 저장합니다.

artifact 구조:

```text
out_dir/
  array_sample_parquet_manifest.json
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  raw_samples/
    sample_000000000000.parquet
  raw_trace_index/
    sample_000000000000.parquet
  sample_parts/
    part_000000.parquet
  trace_index_parts/
    part_000000.parquet
```

## Build Behavior

builder는 final part를 메모리에 계속 들고 있지 않습니다. sample 하나가 끝날 때 raw point parquet와 raw trace-index parquet를 commit하고, 마지막 compact 단계에서 raw sample 파일들을 final part로 묶습니다.

commit 규칙:

- sample close 시 `.tmp` 파일을 먼저 씁니다.
- raw point와 raw trace-index 파일이 모두 정상 생성되면 최종 `.parquet`로 rename합니다.
- `raw_samples.jsonl`에 commit record를 append합니다.
- resume 시에는 commit log에 있고 실제 파일도 존재하는 sample만 완료로 인정합니다.

part flush 기준은 `target_part_bytes`, `max_part_rows`, `max_part_samples`입니다. 기본적으로 sample 개수보다 추정 byte와 row 수를 우선합니다.

## Config Guide

처음에는 아래 설정만 넣으면 됩니다.

```python
ArraySampleParquetBuildOptions(
    target_part_bytes=128 * 1024 * 1024,
    compression="zstd",
)
```

| option | 기본값 | 설명 |
| --- | --- | --- |
| `target_part_bytes` | 128MB | final part 하나의 목표 크기입니다. part가 너무 많으면 키우고, 한 part가 너무 크면 줄입니다. |
| `max_part_rows` | 10,000,000 | point row 수 기준 안전장치입니다. trace가 매우 길 때 줄일 수 있습니다. |
| `max_part_samples` | 0 | part 하나의 최대 sample 수입니다. 0이면 sample 수로 제한하지 않습니다. |
| `compression` | `"zstd"` | parquet compression입니다. 디버깅/속도 확인은 `"none"`, 저장 용량은 `"zstd"`를 씁니다. |
| `sample_key_col` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `feature_key_col` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |

기본 분할 기준은 sample 수가 아니라 추정 byte와 point row 수입니다. sample마다 trace 길이와 feature 수가 달라질 수 있기 때문입니다.

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
    PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL),
]

options = ArraySampleParquetBuildOptions(
    target_part_bytes=128 * 1024 * 1024,
    max_part_rows=10_000_000,
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
    for sample_id in builder.pending_sample_ids():
        with builder.sample(sample_id=sample_id, skip_if_completed=True) as sample:
            if sample.skipped:
                continue
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
payload = reader.get_traces_json(
    sample_keys=["sample_000001"],
    feature_keys=["feature_a"],
    include_missing=True,
    layout="nested",
)
```

## API Server

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 endpoint:

- `POST /array-sample-parquet/schema`
- `POST /array-sample-parquet/traces`

자세한 포맷, 구현 방식, API 요청/응답은 [docs/array_sample_parquet_format_v1.md](../../docs/array_sample_parquet_format_v1.md)를 참고하십시오.
