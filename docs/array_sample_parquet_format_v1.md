# Array Sample Parquet Format v1

`array_sample_parquet`은 array trace를 viewer/debugging 용도로 쉽게 열어보기 위한 sample-major Parquet 포맷입니다. 빠른 feature-major serving은 기존 `array-binary-shard`가 맡고, 이 포맷은 DuckDB, Polars, PyArrow, parquet-tools 같은 일반 Parquet 도구로 직접 확인하기 쉬운 구조를 우선합니다.

## 전체 구조

최종 artifact는 다음 구조를 가집니다.

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

`sample_parts/*.parquet`은 실제 point payload입니다. row 하나는 trace 하나가 아니라 point 하나입니다.

```text
sample_id  feature_id  point_idx  time  value  ch_step
0          10          0          0.0   10.0   idle
0          10          1          1.0   11.0   run
0          27          0          0.0   30.0   idle
```

`trace_index_parts/*.parquet`은 present trace 목록입니다.

```text
sample_id  feature_id  trace_len
0          10          2
0          27          1
1          10          0
```

long point rows만 있으면 `trace_len=0`인 empty trace와 아예 없는 missing trace를 구분할 수 없습니다. 그래서 모든 present trace는 반드시 trace index에 기록합니다. point row는 `trace_len > 0`인 trace에만 존재합니다.

## 파일 역할

`array_sample_parquet_manifest.json`은 reader/API server의 entrypoint입니다. metadata 경로, point schema, point part 목록, trace index part 목록이 들어갑니다.

`sample_parts/*.parquet`은 long-format point payload입니다. 각 row는 `(sample_id, feature_id, point_idx)` 하나의 point입니다.

`trace_index_parts/*.parquet`은 present trace index입니다. empty trace는 여기에 `trace_len=0`으로만 기록되고, point part에는 row가 없습니다.

`sample_meta.parquet`은 dense `sample_id`와 선택적 `sample_key`를 담습니다. `sample_keys` 조회를 쓰려면 key column이 필요합니다.

`feature_meta.parquet`은 dense `feature_id`와 선택적 `feature_key`를 담습니다. `feature_keys` 조회를 쓰려면 key column이 필요합니다.

`raw_samples/*.parquet`와 `raw_trace_index/*.parquet`는 sample별 resume-safe 중간 산출물입니다. sample 하나가 정상 종료될 때만 `.tmp`에서 최종 `.parquet`로 rename됩니다.

`raw_samples.jsonl`은 append-only raw sample commit log입니다. resume 시에는 이 로그에 기록되고 실제 파일도 존재하는 sample만 completed sample로 인정합니다.

`raw_state.json`은 build session snapshot입니다. 옵션, point schema, feature key 목록, compact 완료 여부와 manifest path가 들어갑니다.

## Parquet Schema

### Point Part

```text
sample_id  int64
feature_id int32
point_idx  int32
<point column> typed primitive
```

예를 들어 point schema가 `time: float64`, `value: float64`, `ch_step: categorical string`이면 point part schema는 다음과 같습니다.

```text
sample_id  int64
feature_id int32
point_idx  int32
time       float64
value      float64
ch_step    string
```

### Trace Index Part

```text
sample_id  int64
feature_id int32
trace_len  int32
```

각 present trace마다 trace index row가 정확히 하나 있어야 합니다. 같은 `(sample_id, feature_id)`에 대해 point part에는 `point_idx = 0..trace_len-1` row가 있어야 합니다.

## Point Schema Type

지원하는 logical/storage 조합은 다음과 같습니다.

| logical_type | storage_type |
| --- | --- |
| `continuous` | `float64` |
| `integer` | `int32`, `int64`, `uint32`, `uint64` |
| `categorical` | `string` |
| `timestamp_ns` | `int64` |
| `timedelta_ns` | `int64` |

categorical은 final parquet에도 string primitive로 저장합니다. 별도 `categorical_dictionaries/*.json` sidecar나 global code mapping은 만들지 않습니다. 반복 문자열의 저장 최적화는 Parquet writer의 dictionary/RLE encoding에 맡깁니다.

point-level null은 지원하지 않습니다. `float64`에는 IEEE `NaN`을 넣을 수 있지만 Parquet null은 아닙니다. categorical string에도 null을 넣을 수 없습니다.

## Missing And Empty

missing trace는 trace index row가 없는 상태입니다.

empty trace는 trace index row가 있고 `trace_len=0`인 상태입니다. 이 경우 point part에는 해당 `(sample_id, feature_id)` row가 없습니다.

API에서 `include_missing=true`와 명시적 feature 목록을 주면, trace index에 없는 sample-feature 조합을 `present=false`로 채웁니다.

## Build Algorithm

builder는 sample-major ingest를 받지만 최종 part에 바로 쓰지 않습니다. sample 하나가 끝날 때 raw sample parquet를 확정하고, 마지막에 raw 파일들을 compact해서 최종 part를 만듭니다.

1. `open_session(...)`이 `out_dir`을 초기화하거나 기존 `raw_state.json`을 읽어 resume합니다.
2. `sample_meta.parquet`과 `feature_meta.parquet`을 artifact 안으로 복사하거나 생성합니다.
3. 사용자는 `builder.status().pending_sample_ids`를 보고 아직 완료되지 않은 sample을 씁니다. 순차 예제에서는 `next_expected_sample_id`를 써도 됩니다.
4. `sample.add_trace(...)`는 point column 이름과 길이를 검증합니다.
5. categorical column은 code 변환 없이 문자열 배열로 정규화합니다.
6. sample 종료 시 `raw_samples/sample_*.parquet.tmp`와 `raw_trace_index/sample_*.parquet.tmp`를 씁니다.
7. 두 raw 파일이 모두 정상 생성되면 final `.parquet`로 rename하고 `raw_samples.jsonl`에 commit record를 append합니다.
8. `finish()` 또는 `compact()`는 raw 파일들을 part 크기 기준으로 묶고 final parquet를 씁니다. Java 구현은 raw sample을 쓸 때 trace 목록을 먼저 정렬해 두므로 compact 단계에서 별도 SQL `ORDER BY` 없이 raw 파일 목록 순서를 그대로 유지합니다.
9. 모든 final part가 생성되면 manifest를 씁니다.

flush 조건은 `max_part_samples`, `max_part_rows`, `target_part_bytes`로 제어합니다. long format에서 `max_part_rows`는 point row 기준이며 기본값은 `10_000_000`입니다.

최종 point part의 물리 순서는 `(sample_id, feature_id, point_idx)`입니다. raw 파일도 같은 long row 모델을 사용하므로 parquet-tools, DuckDB, Polars로 중간 산출물을 직접 확인할 수 있습니다.

## Raw Builder

Python의 `ArraySampleParquetRawDatasetBuilder`와 Java의 `ArraySampleParquetDatasetBuilder`는 out-of-order/sample-parallel ingest를 위한 raw builder입니다. worker는 sample 단위 raw parquet를 만들고, supervisor가 `compact()`로 최종 `sample_parts`와 `trace_index_parts`를 생성합니다.

raw builder의 sample별 중간 파일도 final과 같은 long 의미 모델을 사용합니다.

```text
raw_samples/sample_000000000017.parquet
sample_id  int64
feature_id int32
point_idx  int32
<point column> primitive

raw_trace_index/sample_000000000017.parquet
sample_id  int64
feature_id int32
trace_len  int32
```

raw와 final 모두 categorical을 string으로 저장하므로 compact 단계에서 label scan, dictionary 생성, code 변환을 하지 않습니다. compact는 raw sample 파일들을 part 크기 기준으로 묶고 final parquet을 씁니다. raw 파일은 sample close 시점에 이미 `(sample_id, feature_id, point_idx)` 순서를 만족해야 하며, Java 구현은 이를 `ArraySampleParquetOrderChecks`로 검증할 수 있습니다.

Python public API는 다음 형태입니다.

```python
builder = ArraySampleParquetRawDatasetBuilder.open_session(...)
pending = builder.pending_sample_ids()

with builder.sample(sample_id=17, skip_if_completed=True) as sample:
    if not sample.skipped:
        sample.add_trace(feature_key="feature_a", columns={...})

manifest_path = builder.compact()
```

Java public API는 같은 개념을 기존 class name으로 제공합니다.

```java
ArraySampleParquetBuildSessionStatus status = builder.status();
for (Long sampleId : status.pendingSampleIds) {
    try (ArraySampleParquetSampleContext sample = builder.sample(sampleId.longValue(), true)) {
        if (!sample.skipped) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
}
String manifestPath = builder.compact();
```

compact 결과는 일반 reader가 읽는 long-format artifact입니다.

### Raw Builder 성능 특성

raw 파일을 long point row로 저장하면 sample별 raw write 단계에서 `point_idx`, `sample_id`, `feature_id` column을 point 수만큼 더 써야 하므로 wide/list raw보다 약간 느리고 중간 파일도 조금 커질 수 있습니다. 대신 raw와 final의 의미 모델이 같아져 중간 파일을 parquet-tools나 DuckDB로 열었을 때 final point part와 같은 방식으로 디버깅할 수 있습니다.

20 samples, 1200 features, trace_len 950, `time/value float64 + ch_step categorical string` 기준 Python raw builder 로컬 측정에서는 writer-only raw write가 약 `5.81s`, compact-only가 약 `3.87s`였습니다. raw point files는 약 `150.24MB`, raw trace index는 약 `0.10MB`, final `sample_parts`는 약 `159.34MB`, final trace index는 약 `0.02MB`, part 수는 3개였습니다. reader가 Python list로 재조립한 조회 시간은 feature 하나 x 전체 sample 약 `130.6ms`, sample 하나 x 전체 feature 약 `709.3ms`, 16 samples x 64 features 약 `451.9ms`였습니다.

Java raw builder도 같은 포맷을 만들며, point row 단위 `DuckDBAppender` 대신 Java 8 호환 Apache Arrow vector batch를 DuckDB `registerArrowStream(...)`으로 넘깁니다. sample close 직전에 trace 목록을 `(sample_id, feature_id)` 순서로 정렬하고, raw write와 compact 단계에서는 DuckDB SQL `ORDER BY` 없이 `COPY ... TO parquet`를 수행합니다.

같은 크기 로컬 측정에서 Java 전체 build는 약 `8.2s ~ 10.0s`, raw sample close/write 합계는 약 `4.9s ~ 6.2s`, compact는 약 `2.9s ~ 3.3s`였습니다. raw point files는 약 `166.47MB`, final `sample_parts`는 약 `166.38MB`, part 수는 3개였습니다. raw sample, raw trace index, final sample part, final trace index 모두 물리 row 정렬 검사를 통과했습니다.

## Reader Algorithm

reader는 manifest를 읽고 sample 범위가 겹치는 candidate part만 선택합니다.

1. `sample_ids` 또는 `sample_keys` 중 정확히 하나를 받습니다.
2. `feature_ids` 또는 `feature_keys`는 선택 사항입니다.
3. candidate `trace_index_parts`를 읽어 present trace 목록과 `trace_len`을 얻습니다.
4. candidate `sample_parts`를 읽고 `sample_id`, `feature_id`로 filter한 뒤 `(sample_id, feature_id)`별로 `point_idx` 순서의 list를 재구성합니다.
5. trace index에는 있지만 point row가 없는 trace는 `trace_len=0` empty trace로 반환합니다.
6. `include_missing=true`면 명시된 sample-feature 조합 중 trace index에 없는 것을 `present=false`로 추가합니다.

## API Server

`python/scripts/serve_array_api.py`의 array sample parquet endpoint는 manifest path를 받아 같은 reader를 사용합니다.

### `POST /array-sample-parquet/schema`

요청:

```json
{
  "manifest_path": "data/array_sample_parquet/array_sample_parquet_manifest.json",
  "include_dictionaries": true
}
```

응답에는 point schema, sample/feature 수, part 수가 포함됩니다. array sample parquet에는 categorical sidecar dictionary가 없으므로 `categorical_dictionaries`는 `null`입니다.

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

응답은 trace 단위 layout입니다. 내부 저장이 long format이어도 API 응답은 `columns: {name: [...]}` 형태로 재조립됩니다. categorical column은 이미 문자열 list로 반환됩니다.
