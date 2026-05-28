# Array Sample Parquet Format v1

`array_sample_parquet`은 array trace를 **sample-major Parquet**으로 저장하는 포맷입니다. 기존 `array-binary-shard`가 feature-major custom binary로 빠른 serving을 목표로 한다면, 이 포맷은 DuckDB, Polars, PyArrow, parquet-tools 같은 일반 도구로 쉽게 열어볼 수 있는 viewer/debugging 용도를 우선합니다.

핵심 차이는 다음과 같습니다.

- 기존 array binary shard: feature 단위 조회와 random access가 빠르지만, 파일을 직접 디버깅하기 어렵습니다.
- array sample parquet: sample 단위로 데이터가 모이고 Parquet list column을 쓰므로 구조가 투명하지만, feature 하나를 모든 sample에서 훑는 조회는 느릴 수 있습니다.
- missing trace는 row 부재로 표현하고, empty trace는 `trace_len=0`인 row로 표현합니다.
- part 크기는 sample 개수가 아니라 `target_part_bytes` 기반으로 자동 결정합니다.

## 전체 구조

입력은 sample 순서대로 들어옵니다. sample 하나 안에는 여러 feature trace가 있을 수 있고, trace 하나는 point column list들의 묶음입니다.

```text
sample 0
  feature 10 -> time[], value[], ch_step[]
  feature 27 -> time[], value[], ch_step[]
sample 1
  feature 10 -> time[], value[], ch_step[]
...
```

Parquet part의 row 하나는 `(sample_id, feature_id)` trace 하나입니다.

```text
sample_id  feature_id  trace_len  time[]       value[]        ch_step[]
0          10          3          [0,1,2]      [10,11,12]    [1,2,2]
0          27          2          [0,1]        [30,31]       [3,3]
1          10          0          []           []            []
```

최종 artifact는 다음 구조를 가집니다.

```text
out_dir/
  array_sample_parquet_manifest.json
  state.json
  parts.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_parts/
    part_000000.parquet
    part_000001.parquet
  categorical_dictionaries/
    ch_step.json
```

`sample_parts/part_*.parquet` 하나에는 여러 sample의 trace row가 들어갑니다. part는 sample 중간에서 닫히지 않고, sample 하나가 완전히 끝난 뒤에만 commit 여부를 판단합니다. 그래서 재시작 시 사용자는 `status().next_expected_sample_id` 또는 Java의 `status().nextExpectedSampleId`부터 다시 넣으면 됩니다.

## 파일 역할

`array_sample_parquet_manifest.json`은 최종 dataset manifest입니다. reader와 API server는 이 파일 하나를 entrypoint로 사용합니다. metadata 경로, point schema, part 목록, categorical dictionary 경로가 여기에 들어갑니다.

`sample_parts/*.parquet`은 실제 trace payload입니다. row 하나가 `(sample_id, feature_id)` trace 하나이고, point column은 Parquet list column입니다.

`sample_meta.parquet`은 dense `sample_id`와 선택적인 `sample_key`를 담습니다. API에서 `sample_keys`로 조회하려면 `sample_key` 같은 key column이 필요합니다.

`feature_meta.parquet`은 dense `feature_id`와 선택적인 `feature_key`를 담습니다. API에서 `feature_keys`로 조회하려면 `feature_key` 같은 key column이 필요합니다.

`categorical_dictionaries/*.json`은 categorical point column의 code -> label mapping입니다. Parquet에는 `uint32` code만 저장하고, reader/API에서 `decode_categorical=true`일 때 label로 복원합니다.

`parts.jsonl`은 append-only commit log입니다. resume 시에는 이 로그에 기록된 part만 committed part로 인정합니다. `.parquet.tmp`는 아직 commit되지 않은 파일이므로 resume 과정에서 삭제됩니다.

`state.json`은 build session snapshot입니다. 옵션, point schema, feature key 목록, categorical label 상태, 마지막 committed sample, 다음 ingest sample 위치가 들어갑니다.

## Parquet Row Schema

part parquet의 row 하나는 trace 하나입니다.

```text
sample_id  int64
feature_id int32
trace_len  int32
<point column> list<typed value>
```

예를 들어 point schema가 `time: float64`, `value: float64`, `ch_step: categorical uint32`이면 row schema는 다음과 같습니다.

```text
sample_id  int64
feature_id int32
trace_len  int32
time       list<float64>
value      list<float64>
ch_step    list<uint32>
```

모든 point column list 길이는 반드시 `trace_len`과 같아야 합니다. point-level null은 지원하지 않습니다.

## Point Schema Type

지원하는 logical/storage 조합은 기존 array v3 point schema와 맞춥니다.

| logical_type | storage_type |
| --- | --- |
| `continuous` | `float64` |
| `integer` | `int32`, `int64`, `uint32`, `uint64` |
| `categorical` | `uint32` |
| `timestamp_ns` | `int64` |
| `timedelta_ns` | `int64` |

`timestamp_ns`와 `timedelta_ns`는 Parquet 물리 저장으로는 `int64` nanosecond 값을 저장합니다. 표시 형식 변환은 reader/API layer에서 수행합니다.

## Missing And Empty

missing trace는 row가 없는 상태입니다. 예를 들어 `(sample_id=5, feature_id=10)` row가 없으면 그 trace는 missing입니다.

empty trace는 row가 있고 `trace_len=0`이며 모든 list column이 빈 list인 상태입니다. 즉 "존재하지만 point가 없다"를 표현합니다.

point-level null은 지원하지 않습니다. `float64` column에는 IEEE `NaN`을 넣을 수 있지만, 이것은 Parquet null이 아닙니다. categorical은 code `0`을 missing/unknown 예약값으로 두고, 실제 label은 code `1..N`에 배정합니다.

## Build Algorithm

builder는 sample-major ingest를 받지만, part 파일을 만들 때는 전체 part를 메모리에 들고 있지 않습니다. trace row를 받는 즉시 `.parquet.tmp`에 streaming append하고, 작은 row batch만 메모리에 유지합니다.

1. `open_session(...)`이 호출되면 `out_dir`을 초기화하거나 기존 `state.json`을 읽어 resume합니다.
2. `sample_meta.parquet`은 artifact 안으로 복사합니다.
3. feature metadata는 `feature_meta_path`가 있으면 복사하고, 없으면 ingest 중 등장한 `feature_key` 순서로 나중에 생성합니다.
4. 사용자는 `builder.status().next_expected_sample_id`부터 sample을 순서대로 처리합니다.
5. `with builder.sample(sample_id=...)` 또는 Java의 `try (sample = builder.sample(...))`가 sample 경계를 엽니다.
6. `add_trace(...)`는 point column 이름과 길이를 검증하고, categorical label을 code로 인코딩합니다.
7. 첫 trace가 들어오면 `sample_parts/part_XXXXXX.parquet.tmp` writer를 엽니다.
8. trace row는 바로 writer로 전달됩니다. Python은 `pyarrow.parquet.ParquetWriter`, Java는 `parquet-hadoop`의 `ExampleParquetWriter`를 사용합니다.
9. sample이 끝나면 pending sample 통계를 갱신하고 flush 조건을 검사합니다.
10. flush 조건을 만족하면 writer를 close하여 Parquet footer를 완성하고 `.parquet.tmp`를 `.parquet`으로 rename합니다.
11. commit record를 `parts.jsonl`에 append하고, `state.json`을 갱신합니다.
12. `finish()`는 남은 part를 commit하고, feature meta와 categorical dictionary를 쓴 뒤 최종 manifest를 생성합니다.

part flush 조건은 다음 순서로 판단합니다.

- `max_part_samples > 0`이고 pending sample 수가 이 값 이상이면 flush합니다.
- `part_rows >= max_part_rows`이면 flush합니다.
- 추정 payload byte가 `target_part_bytes` 이상이면 flush합니다.

기본값은 `target_part_bytes = 128MB`, `max_part_rows = 100000`, `max_part_samples = 0`입니다. `max_part_samples=0`은 sample 수 cap을 쓰지 않는다는 뜻입니다. 즉 기본 동작은 sample 개수 기준이 아니라 byte 기준입니다.

## Resume And Failure Semantics

durable checkpoint 단위는 committed part입니다. committed part는 `parts.jsonl`에 기록되고, final `.parquet` 파일이 존재하는 part입니다.

중간에 프로세스가 종료되면 다음 규칙을 따릅니다.

- `.parquet.tmp`는 uncommitted 파일이므로 resume 시 삭제합니다.
- `parts.jsonl`에 없는 part는 reader와 manifest에서 무시합니다.
- `state.json`과 `parts.jsonl`을 기준으로 `next_expected_sample_id = last_committed_sample_id + 1`을 계산합니다.
- 사용자는 그 sample부터 다시 데이터를 수집해서 넣으면 됩니다.
- commit되지 않은 sample은 재실행되어야 합니다.
- categorical dictionary label 상태는 `state.json`과 마지막 part commit record의 label snapshot으로 복구합니다.

sample context 안에서 예외가 발생하면 현재 uncommitted part 전체를 버리고 마지막 committed sample 다음 위치로 cursor를 되돌립니다. Parquet writer는 이미 쓴 row만 개별 rollback할 수 없으므로, uncommitted part를 통째로 버리는 방식이 가장 단순하고 안전합니다.

## Reader Algorithm

reader는 manifest를 읽고 metadata와 part 목록을 로드합니다.

조회 흐름은 다음과 같습니다.

1. `sample_ids` 또는 `sample_keys` 중 정확히 하나를 받습니다.
2. `feature_ids` 또는 `feature_keys`는 둘 중 하나만 허용합니다. 둘 다 없으면 요청 sample 안에 실제 존재하는 trace를 모두 반환합니다.
3. sample key와 feature key는 metadata parquet로 id에 매핑합니다.
4. part 목록의 `first_sample_id`, `last_sample_id` 범위로 candidate part를 줄입니다.
5. candidate part parquet를 scan하고 `sample_id`, `feature_id` 조건으로 filter합니다.
6. categorical decode가 요청되면 dictionary JSON을 읽어 code를 label로 바꿉니다.
7. `include_missing=true`이면 요청한 sample-feature 조합 중 row가 없는 trace도 `present=false`로 채웁니다.

Python reader는 Polars/PyArrow 기반이고, Java reader는 DuckDB와 Parquet list column 읽기를 사용합니다.

## API Server

기존 `python/scripts/serve_array_api.py`에 sample-major Parquet 전용 endpoint가 있습니다.

### `POST /array-sample-parquet/schema`

요청:

```json
{
  "manifest_path": "data/array_sample_parquet/array_sample_parquet_manifest.json",
  "include_dictionaries": true
}
```

응답:

```json
{
  "manifest_path": ".../array_sample_parquet_manifest.json",
  "format": "array-sample-parquet",
  "version": 1,
  "n_samples": 1000,
  "n_features": 1024,
  "sample_key_col": "sample_key",
  "feature_key_col": "feature_key",
  "part_count": 8,
  "point_schema": [
    {"name": "time", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "value", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "ch_step", "storage_type": "uint32", "logical_type": "categorical", "dictionary_path": "..."}
  ],
  "categorical_dictionaries": {
    "ch_step": {"1": "idle", "2": "run"}
  }
}
```

### `POST /array-sample-parquet/traces`

`sample_ids` 또는 `sample_keys` 중 정확히 하나를 줘야 합니다. `feature_ids`와 `feature_keys`는 둘 중 하나만 허용합니다. feature 조건을 생략하면 요청 sample에 실제 존재하는 trace만 반환합니다.

요청:

```json
{
  "manifest_path": "data/array_sample_parquet/array_sample_parquet_manifest.json",
  "sample_keys": ["sample_000001", "sample_000002"],
  "feature_keys": ["feature_a", "feature_b"],
  "include_missing": true,
  "decode_categorical": true,
  "layout": "nested",
  "max_traces": 10000
}
```

`nested` 응답:

```json
{
  "manifest_path": ".../array_sample_parquet_manifest.json",
  "layout": "nested",
  "sample_count": 2,
  "trace_count": 4,
  "samples": [
    {
      "sample_id": 1,
      "sample_key": "sample_000001",
      "traces": [
        {
          "feature_id": 10,
          "feature_key": "feature_a",
          "present": true,
          "trace_len": 3,
          "columns": {
            "time": [0.0, 1.0, 2.0],
            "value": [10.0, 11.0, 12.0],
            "ch_step": ["idle", "run", "run"]
          }
        }
      ]
    }
  ]
}
```

`flat` 응답은 `traces` 배열에 sample/feature 정보를 모두 포함한 row 형태로 반환합니다.

```json
{
  "layout": "flat",
  "trace_count": 1,
  "traces": [
    {
      "sample_id": 1,
      "sample_key": "sample_000001",
      "feature_id": 10,
      "feature_key": "feature_a",
      "present": true,
      "trace_len": 3,
      "columns": {
        "time": [0.0, 1.0, 2.0],
        "value": [10.0, 11.0, 12.0],
        "ch_step": ["idle", "run", "run"]
      }
    }
  ]
}
```

## Python Usage

```python
from fs.array_sample_parquet import (
    ArraySampleParquetBuildOptions,
    ArraySampleParquetDatasetBuilder,
    open_array_sample_parquet,
)
from fs.types import LogicalType, PointColumnSpec, StorageType

point_schema = [
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
    point_schema,
    feature_meta_path="data/feature_meta.parquet",
    options=options,
) as builder:
    start = builder.status().next_expected_sample_id
    for sample_id in range(start, n_samples):
        with builder.sample(sample_id=sample_id) as sample:
            sample.add_trace(
                feature_key="feature_a",
                columns={
                    "time": [0.0, 1.0, 2.0],
                    "value": [10.0, 11.0, 12.0],
                    "ch_step": ["idle", "run", "run"],
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

## Java Usage

```java
ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
options.targetPartBytes = 128L * 1024L * 1024L;
options.maxPartRows = 100000;
options.maxPartSamples = 0;
options.compression = "zstd";

try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
        "data/array_sample_parquet",
        "data/sample_meta.parquet",
        pointSchema,
        "data/feature_meta.parquet",
        options)) {
    long start = builder.status().nextExpectedSampleId;
    for (long sampleId = start; sampleId < nSamples; sampleId++) {
        try (ArraySampleParquetSampleContext sample = builder.sample(sampleId)) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
    String manifestPath = builder.finish();
}
```

## Implementation Map

Python implementation:

- `python/fs/array_sample_parquet/builder.py`: resumable session, sample boundary, commit log, state 관리
- `python/fs/array_sample_parquet/parquet_io.py`: PyArrow 기반 streaming part writer
- `python/fs/array_sample_parquet/reader.py`: manifest 기반 trace 조회와 JSON layout 변환
- `python/fs/array_sample_parquet/manifest.py`: manifest/options/status model
- `python/scripts/serve_array_api.py`: HTTP API endpoint

Java implementation:

- `java/src/fs/io/ArraySampleParquets.java`: public facade
- `java/src/fs/io/array_sample_parquet/ArraySampleParquetDatasetBuilder.java`: resumable session, sample boundary, commit log, state 관리
- `java/src/fs/io/array_sample_parquet/ArraySampleParquetPartWriter.java`: parquet-hadoop 기반 streaming part writer
- `java/src/fs/io/array_sample_parquet/ArraySampleParquetReader.java`: DuckDB 기반 trace 조회
- `java/src/fs/model/array_sample_parquet/*.java`: manifest/options/status/trace model

패키지 문서:

- Python wheel: `packages/array_sample_parquet/README.md`
- Java jar: `packages/array_sample_parquet_java/README.md`

## Performance Guidance

이 포맷은 디버깅과 sample 중심 viewer에 맞춘 포맷입니다.

- sample 하나의 여러 feature trace를 보는 조회는 sample-major layout 덕분에 유리합니다.
- feature 하나를 모든 sample에서 훑는 조회는 여러 part를 scan해야 하므로 기존 `array-binary-shard`보다 느립니다.
- feature-major serving이나 대량 random access가 중요하면 기존 custom binary shard를 사용해야 합니다.
- `target_part_bytes`를 너무 작게 잡으면 part 수가 늘어 metadata/scan overhead가 커집니다.
- `target_part_bytes`를 너무 크게 잡으면 sample 범위 pruning이 거칠어지고 단일 part scan 비용이 커집니다.
- 기본 128MB는 일반적인 Parquet scan과 파일 수 사이의 균형을 위한 보수적 기본값입니다.
