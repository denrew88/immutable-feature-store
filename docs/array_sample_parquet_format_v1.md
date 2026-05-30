# Array Sample Parquet Format v1

`array_sample_parquet`은 array trace를 viewer/debugging 용도로 쉽게 열어보기 위한 sample-major Parquet 포맷입니다. 빠른 feature-major serving은 기존 `array-binary-shard`가 맡고, 이 포맷은 DuckDB, Polars, PyArrow, parquet-tools 같은 일반 Parquet 도구로 직접 확인하기 쉬운 구조를 우선합니다.

## 전체 구조

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

`parts.jsonl`은 append-only commit log입니다. resume 시에는 이 로그에 기록된 part만 committed part로 인정합니다.

`state.json`은 build session snapshot입니다. 옵션, point schema, feature key 목록, 마지막 committed sample, 다음 ingest sample 위치가 들어갑니다.

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

builder는 sample-major ingest를 받습니다. part commit은 sample 경계에서만 일어나므로 중간 실패 후에는 마지막 committed sample 다음부터 다시 넣으면 됩니다.

1. `open_session(...)`이 `out_dir`을 초기화하거나 기존 `state.json`을 읽어 resume합니다.
2. `sample_meta.parquet`과 `feature_meta.parquet`을 artifact 안으로 복사하거나 생성합니다.
3. 사용자는 `builder.status().next_expected_sample_id`부터 sample을 순서대로 넣습니다.
4. `sample.add_trace(...)`는 point column 이름과 길이를 검증합니다.
5. categorical column은 code 변환 없이 문자열 배열로 정규화합니다.
6. trace 하나를 쓰면 trace index writer에 `(sample_id, feature_id, trace_len)`이 기록됩니다.
7. `trace_len > 0`이면 point part writer에 `point_idx`별 row가 펼쳐져 기록됩니다.
8. flush 조건이 만족되면 `.parquet.tmp`를 닫고 final `.parquet`로 rename한 뒤 `parts.jsonl`과 `state.json`을 갱신합니다.
9. `finish()`는 남은 part를 commit하고 manifest를 씁니다.

flush 조건은 `max_part_samples`, `max_part_rows`, `target_part_bytes`로 제어합니다. long format에서 `max_part_rows`는 point row 기준이며 기본값은 `10_000_000`입니다.

최종 point part의 물리 순서는 `(sample_id, feature_id, point_idx)`입니다. sequential builder는 sample 종료 시 sample 내부 trace를 `feature_id` 기준으로 정렬한 뒤 기록합니다.

## Raw Builder

`ArraySampleParquetRawDatasetBuilder`는 out-of-order/sample-parallel ingest를 위한 builder입니다. worker는 sample 단위 raw parquet를 만들고, supervisor가 `compact()`로 최종 `sample_parts`와 `trace_index_parts`를 생성합니다.

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

raw와 final 모두 categorical을 string으로 저장하므로 compact 단계에서 label scan, dictionary 생성, code 변환을 하지 않습니다. compact는 raw sample 파일들을 part 크기 기준으로 묶고, `(sample_id, feature_id, point_idx)` 순서로 정렬한 뒤 final parquet을 씁니다.

현재 public API는 다음 형태입니다.

```python
builder = ArraySampleParquetRawDatasetBuilder.open_session(...)
pending = builder.pending_sample_ids()

with builder.sample(sample_id=17, skip_if_completed=True) as sample:
    if not sample.skipped:
        sample.add_trace(feature_key="feature_a", columns={...})

manifest_path = builder.compact()
```

compact 결과는 일반 `ArraySampleParquetDatasetBuilder`와 같은 long-format artifact입니다.

### Raw Builder 성능 특성

raw 파일을 long point row로 저장하면 sample별 raw write 단계에서 `point_idx`, `sample_id`, `feature_id` column을 point 수만큼 더 써야 하므로 wide/list raw보다 약간 느리고 중간 파일도 조금 커질 수 있습니다. 대신 raw와 final의 의미 모델이 같아져 중간 파일을 parquet-tools나 DuckDB로 열었을 때 final point part와 같은 방식으로 디버깅할 수 있습니다.

20 samples, 1200 features, trace_len 950, `time/value float64 + ch_step categorical string` 기준 로컬 측정에서는 raw write가 약 `10.42s`, compact가 약 `7.70s`였습니다. raw point files는 약 `150.24MB`, raw trace index는 약 `0.10MB`, final `sample_parts`는 약 `159.34MB`, final trace index는 약 `0.02MB`, part 수는 3개였습니다. reader가 Python list로 재조립한 조회 시간은 feature 하나 x 전체 sample 약 `130.6ms`, sample 하나 x 전체 feature 약 `709.3ms`, 16 samples x 64 features 약 `451.9ms`였습니다.

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
