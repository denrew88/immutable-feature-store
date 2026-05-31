# Array Sample Parquet Format v1

`array_sample_parquet`은 array trace를 viewer/debugging 용도로 쉽게 열어보기 위한 sample-major long Parquet 포맷입니다. 빠른 feature-major serving은 기존 `array-binary-shard`가 맡고, 이 포맷은 DuckDB, Polars, PyArrow, parquet-tools 같은 일반 Parquet 도구로 직접 확인하기 쉬운 구조를 우선합니다.

## 전체 구조

최종 artifact는 다음 구조를 갖습니다.

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
    part_000001.parquet
  trace_index_parts/
    part_000000.parquet
    part_000001.parquet
```

`raw_*` 파일들은 resume 가능한 중간 산출물이고, `sample_parts/`와 `trace_index_parts/`가 reader/API server가 읽는 최종 산출물입니다.

## 파일 역할

- `array_sample_parquet_manifest.json`: reader/API server의 entrypoint입니다. metadata 경로, point schema, part 목록이 들어 있습니다.
- `sample_meta.parquet`: dense `sample_id`와 선택적 `sample_key`를 담습니다.
- `feature_meta.parquet`: dense `feature_id`와 선택적 `feature_key`를 담습니다.
- `raw_samples/*.parquet`: sample 하나의 long-format point rows입니다.
- `raw_trace_index/*.parquet`: sample 하나의 present trace index입니다.
- `raw_samples.jsonl`: append-only raw sample commit log입니다.
- `raw_state.json`: build option, point schema, 완료 manifest 여부를 담는 session snapshot입니다.
- `sample_parts/*.parquet`: final point rows입니다.
- `trace_index_parts/*.parquet`: final present trace index입니다.

## Parquet Schema

### Point Part

```text
sample_id  int64
feature_id int32
point_idx  int32
<point column> typed primitive
```

예를 들어 point schema가 `time: float64`, `value: float64`, `ch_step: categorical string`이면 다음과 같습니다.

```text
sample_id  int64
feature_id int32
point_idx  int32
time       float64
value      float64
ch_step    string
```

row 하나는 trace 하나가 아니라 point 하나입니다. 같은 `(sample_id, feature_id)` 안에서 `point_idx`가 `0..trace_len-1` 순서로 증가합니다.

### Trace Index Part

```text
sample_id  int64
feature_id int32
trace_len  int32
```

present trace마다 trace index row가 하나 있습니다. `trace_len > 0`이면 point part에 같은 `(sample_id, feature_id)`와 해당 point rows가 있어야 합니다. `trace_len=0`이면 empty trace이며 point row는 없습니다.

## Point Schema Type

지원 logical/storage 조합:

| logical_type | storage_type |
| --- | --- |
| `continuous` | `float64` |
| `integer` | `int32`, `int64`, `uint32`, `uint64` |
| `categorical` | `string` |
| `timestamp_ns` | `int64` |
| `timedelta_ns` | `int64` |

categorical은 raw와 final 모두 string primitive column으로 저장합니다. 별도 `categorical_dictionaries/*.json`이나 global code mapping은 만들지 않습니다. 반복 문자열 압축은 Parquet writer의 dictionary/RLE encoding에 맡깁니다.

point-level null은 지원하지 않습니다. `float64`에는 IEEE `NaN`을 넣을 수 있지만 Parquet null과는 다릅니다.

## Missing과 Empty

- missing trace: trace index row가 없습니다.
- empty trace: trace index row가 있고 `trace_len=0`입니다.
- non-empty trace: trace index row가 있고 point part에 `trace_len`개의 point row가 있습니다.

API에서 `include_missing=true`와 명시적인 feature 목록을 주면 trace index에 없는 sample-feature 조합을 `present=false`로 채워 반환합니다.

## Build Algorithm

표준 builder는 sample별 raw parquet를 먼저 만들고 마지막에 compact합니다.

1. `open_session(...)`이 `out_dir`을 초기화하거나 기존 `raw_state.json`을 읽어 resume합니다.
2. `sample_meta.parquet`와 `feature_meta.parquet`를 artifact 안으로 복사하거나 생성합니다.
3. 사용자는 `builder.pending_sample_ids()` 또는 `builder.status().pending_sample_ids`를 보고 남은 sample을 처리합니다.
4. `builder.sample(...)` context 안에서 `sample.add_trace(...)`를 호출합니다.
5. sample close 시 trace 목록을 `(sample_id, feature_id)` 순서로 정렬하고 point rows는 `(sample_id, feature_id, point_idx)` 순서로 씁니다.
6. 먼저 `.tmp` 파일을 쓰고, raw point와 raw trace-index 파일이 모두 성공하면 최종 `.parquet`로 rename합니다.
7. `raw_samples.jsonl`에 commit record를 append합니다.
8. `finish()` 또는 `compact()`가 raw sample 파일들을 part 크기 기준으로 묶고 final parquet를 씁니다.
9. final part가 모두 생성되면 `array_sample_parquet_manifest.json`을 씁니다.

part flush 기준:

- `target_part_bytes`: 추정 payload byte 기준입니다.
- `max_part_rows`: point row 수 제한입니다.
- `max_part_samples`: sample 수 제한입니다. 기본값 0은 비활성입니다.

기본 기준은 sample 수가 아니라 byte와 row 수입니다. sample마다 trace 길이와 feature 수가 달라질 수 있기 때문입니다.

## Resume와 병렬 처리

sample 하나가 완료되기 전에는 `.tmp` 파일만 존재합니다. 중간에 프로세스가 죽으면 다음 session 시작 시 `.tmp`는 무시하거나 정리하고, commit log와 실제 final parquet 파일이 모두 있는 sample만 완료로 봅니다.

사용 패턴:

1. supervisor가 session을 열고 `pending_sample_ids()`를 읽습니다.
2. pending sample ids를 worker들에게 나눠줍니다.
3. 각 worker는 같은 `out_dir`에 대해 서로 다른 sample id를 `builder.sample(..., skip_if_completed=True)`로 씁니다.
4. 모든 worker가 끝나면 supervisor가 `compact(require_all=True)` 또는 `finish()`를 호출합니다.

## Python 사용 예

```python
from array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
)

point_schema = [
    PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
    PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
    PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL),
]

with ArraySampleParquetDatasetBuilder.open_session(
    out_dir="data/array_sample_parquet",
    sample_meta_path="data/sample_meta.parquet",
    point_schema=point_schema,
    feature_meta_path="data/feature_meta.parquet",
    options=ArraySampleParquetBuildOptions(target_part_bytes=128 * 1024 * 1024),
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
```

## Java 사용 예

```java
ArraySampleParquetBuildSessionStatus status = builder.status();
for (Long sampleId : status.pendingSampleIds) {
    try (ArraySampleParquetSampleContext sample = builder.sample(sampleId.longValue(), true)) {
        if (!sample.skipped) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
}
String manifestPath = builder.finish();
```

## Reader Algorithm

reader는 manifest를 읽고 sample 범위가 겹치는 candidate part만 선택합니다.

1. `sample_ids` 또는 `sample_keys` 중 하나를 받습니다.
2. `feature_ids` 또는 `feature_keys`는 선택 사항입니다.
3. candidate `trace_index_parts`를 읽어 present trace 목록과 `trace_len`을 얻습니다.
4. candidate `sample_parts`를 읽고 `sample_id`, `feature_id`로 filter한 뒤 `(sample_id, feature_id)`별 point arrays를 재구성합니다.
5. trace index에는 있지만 point row가 없는 trace는 `trace_len=0` empty trace로 반환합니다.
6. `include_missing=true`이면 명시된 sample-feature 조합 중 trace index에 없는 것을 `present=false`로 추가합니다.

## Java Builder Value API

Java가 외부 시스템에서 값을 받아 builder에 쓰는 경우, 예제 서버는 `python/scripts/serve_synthetic_value_api.py`를 사용합니다. 이 서버는 최종 shard 조회 API가 아니라, Java builder가 sample별 trace 값을 받아오기 위한 value source 예제입니다.

실행:

```powershell
python python\scripts\serve_synthetic_value_api.py --host 127.0.0.1 --port 8010
```

### `POST /array/traces`

요청은 `sample_id` 또는 `sample_key` 중 하나만, `feature_ids` 또는 `feature_keys` 중 하나만 받습니다. `sample_meta_path`와 `feature_meta_path`는 dense id와 key를 해석하기 위해 필요합니다.

주요 요청 필드:

| field | type | required | 설명 |
| --- | --- | --- | --- |
| `sample_meta_path` | string | yes | `sample_id`와 `sample_key`가 들어 있는 metadata parquet |
| `feature_meta_path` | string | yes | `feature_id`와 `feature_key`가 들어 있는 metadata parquet |
| `sample_id` | int | one of | 조회할 sample dense id |
| `sample_key` | string | one of | 조회할 sample external key |
| `feature_ids` | int[] | one of | 조회할 feature dense id 목록 |
| `feature_keys` | string[] | one of | 조회할 feature external key 목록 |
| `sample_key_col` | string | no | sample key column, 기본값 `sample_key` |
| `feature_key_col` | string | no | feature key column, 기본값 `feature_key` |
| `seed` | int | no | synthetic value 생성 seed |
| `missing_feature_rate` | float | no | missing trace 비율 |
| `empty_trace_rate` | float | no | present지만 길이가 0인 empty trace 비율 |
| `min_trace_len` | int | no | 생성 trace 최소 길이 |
| `max_trace_len` | int | no | 생성 trace 최대 길이 |
| `include_missing` | bool | no | missing trace도 응답에 포함할지 여부 |

요청 예:

```json
{
  "sample_meta_path": "data/sample_meta.parquet",
  "feature_meta_path": "data/feature_meta.parquet",
  "sample_id": 0,
  "feature_ids": [0, 1, 2],
  "seed": 7,
  "include_missing": false,
  "min_trace_len": 24,
  "max_trace_len": 48
}
```

응답은 sample 하나에 대한 trace 목록입니다. `present=false` trace는 `include_missing=true`일 때만 포함됩니다. present trace의 `columns`는 point schema와 같은 이름의 배열을 가집니다.

```json
{
  "sample_id": 0,
  "sample_key": "sample_000000",
  "feature_count": 3,
  "trace_count": 3,
  "point_schema": [
    {"name": "time", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "value", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "ch_step", "storage_type": "string", "logical_type": "categorical"}
  ],
  "traces": [
    {
      "feature_id": 0,
      "feature_key": "feature_000000",
      "present": true,
      "trace_len": 24,
      "columns": {
        "time": [0.0, 0.4347826087],
        "value": [0.12, 0.18],
        "ch_step": ["pre", "pre"]
      }
    }
  ]
}
```

`BuildArraySampleParquetFromValueApiMain`과 jar 예제 `BuildArraySampleParquetFromValueApiWithJarExample`는 이 응답의 `traces`를 순회하면서 `present=true`인 trace만 `sample.addTrace(...)`로 기록합니다.

## API Server

권장 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다.

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

### `POST /array-sample-parquet/schema`

요청:

```json
{
  "manifest_path": "data/array_sample_parquet/array_sample_parquet_manifest.json"
}
```

응답에는 format, version, sample/feature 수, part 수, sample/feature key column, point schema가 포함됩니다. categorical sidecar dictionary가 없으므로 dictionary payload는 반환하지 않습니다.

### `POST /array-sample-parquet/traces`

요청:

```json
{
  "manifest_path": "data/array_sample_parquet/array_sample_parquet_manifest.json",
  "sample_keys": ["sample_000001"],
  "feature_keys": ["feature_a"],
  "include_missing": true,
  "layout": "nested"
}
```

응답은 trace 단위 layout입니다. 저장은 long format이지만 API 응답에서는 `columns: {name: [...]}` 형태로 재조립합니다. categorical column은 이미 string list입니다.
