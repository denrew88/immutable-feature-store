# Array Binary Shard Format v3

이 문서는 array trace serving 전용 custom binary shard 포맷의 **v3 스펙과 현재 구현 규칙**을 설명한다.

v3의 핵심 변화는 다음과 같다.

- `sample_id`, `feature_id`를 dense integer id로 고정한다.
- `time`, `value`를 특수 취급하지 않는다.
- manifest마다 고정된 `point_schema`를 갖고, 그 schema에 맞는 point-level column을 저장한다.
- categorical column은 shard payload 안에 string을 직접 넣지 않고 integer code로 저장한다.
- metadata, dictionary, shard 파일을 한 폴더에 묶어 standalone artifact로 관리한다.

즉 v3는 v2의 빠른 lookup 구조를 유지하면서, point-level column schema를 일반화한 버전이다.

---

## 1. 전체 구조

v3 dataset artifact는 보통 아래처럼 생긴다.

```text
array_binary_dataset_v3/
  array_binary_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  categorical_dictionaries/
    state_code.json
    event_type.json
  array_binary_feature_shards/
    shard_0000.blocks.idx
    shard_0000.blocks.bin
    shard_0001.blocks.idx
    shard_0001.blocks.bin
    ...
```

각 파일의 역할:

- `array_binary_shard_manifest.json`
  - dataset 전체 메타데이터
  - dense id 규칙, schema, shard 목록, dictionary 경로를 정의
- `sample_meta.parquet`
  - sample row order를 정의
  - **row index가 곧 `sample_id`**
- `feature_meta.parquet`
  - feature row order를 정의
  - **row index가 곧 `feature_id`**
- `categorical_dictionaries/*.json`
  - categorical point column의 code-to-label 사전
- `shard_XXXX.blocks.idx`
  - block offset table
- `shard_XXXX.blocks.bin`
  - 실제 block payload

중요한 점:

- manifest 안의 경로는 **manifest 기준 relative path**로 저장한다.
- 따라서 dataset 폴더를 통째로 복사하거나 이동해도 그대로 읽을 수 있다.

---

## 2. Dense Id 와 External Key

v3는 내부 lookup을 위해 dense id를 사용한다.

- `sample_id == sample_meta.parquet`의 row index
- `feature_id == feature_meta.parquet`의 row index

예:

- `sample_meta.parquet`의 0번째 row -> `sample_id = 0`
- `feature_meta.parquet`의 17번째 row -> `feature_id = 17`

외부 시스템과의 연결이 필요하면 metadata에 stable key를 둔다.

- `sample_key`
- `feature_key`

reader와 서버 API는 보통 둘 다 지원한다.

- dense id 기반 조회
  - `feature_id`, `sample_id`
- external key 기반 조회
  - `feature_key`, `sample_key`

권장 규칙:

- `sample_key`는 non-null, unique
- `feature_key`는 non-null, unique

---

## 3. Point Schema

### 3.1 개념

v2는 사실상 아래 두 컬럼만 있는 특수한 schema였다.

- `time: float64[]`
- `value: float64[]`

v3는 이것을 일반화한다.

예:

```text
time       = [t0, t1, t2, ...]            float64
value      = [v0, v1, v2, ...]            float64
phase      = [1, 1, 2, 2, ...]            int32
segment_id = [100, 100, 101, 101, ...]    uint64
state_code = [1, 1, 2, 2, ...]            uint32   # categorical code
event_type = [1, 1, 3, 3, ...]            uint32   # categorical code
ts         = [...]                        int64     # timestamp_ns
dt         = [...]                        int64     # timedelta_ns
```

핵심 규칙:

- 한 manifest 안에서는 `point_schema`가 고정이다.
- manifest가 다르면 `point_schema`가 달라도 된다.
- trace 하나 안의 모든 point column은 같은 `point_count`를 가져야 한다.
- payload 안의 column 순서는 `point_schema` 순서와 정확히 일치한다.

### 3.2 enum 기반 정의

현재 구현은 `storage_type`, `logical_type`를 문자열로 저장하지만, 코드 내부에서는 아래 enum을 권장한다.

- `StorageType`
  - `FLOAT64`
  - `INT32`
  - `INT64`
  - `UINT32`
  - `UINT64`
- `LogicalType`
  - `CONTINUOUS`
  - `INTEGER`
  - `CATEGORICAL`
  - `TIMESTAMP_NS`
  - `TIMEDELTA_NS`

manifest JSON에는 canonical string이 들어간다.

예:

```json
{
  "name": "phase",
  "storage_type": "int32",
  "logical_type": "integer"
}
```

### 3.3 허용 조합

v3 base spec에서 허용하는 `(storage_type, logical_type)` 조합은 아래와 같다.

| logical_type | allowed storage_type |
|---|---|
| `continuous` | `float64` |
| `integer` | `int32`, `int64`, `uint32`, `uint64` |
| `categorical` | `uint32` |
| `timestamp_ns` | `int64` |
| `timedelta_ns` | `int64` |

즉:

- `categorical + float64`는 허용되지 않는다.
- `timestamp_ns + uint32`는 허용되지 않는다.

### 3.4 `time`, `value`는 필수 아님

v3에서는 `time`, `value` 컬럼이 없어도 된다.

---

## 4. Categorical Dictionary

categorical point column은 payload 안에 string을 직접 저장하지 않는다.

예를 들어 `state_code`가 categorical column이면:

- payload 안에는 `uint32[]` code만 저장
- 원래 label은 별도 dictionary JSON에 저장

예:

```json
{
  "name": "state_code",
  "storage_type": "uint32",
  "logical_type": "categorical",
  "dictionary_path": "categorical_dictionaries/state_code.json"
}
```

그리고 `state_code.json`은 이런 형태를 갖는다.

```json
{
  "column": "state_code",
  "items": [
    {"code": 1, "label": "OK"},
    {"code": 2, "label": "WARN"},
    {"code": 3, "label": "FAIL"}
  ]
}
```

권장 규칙:

- `0`은 null / unknown / missing 용도로 예약
- 실제 category는 `1..N`

중요:

- categorical column 하나당 dictionary file 하나
- `state_code.json`과 `event_type.json`은 서로 독립적인 예시다
- schema에 `state_code`만 있으면 `state_code.json`만 있으면 된다

---

## 5. Builder 와 Ingestion

### 5.1 권장 public API

현재 구현 기준으로는 sample-major parquet를 사용자가 직접 조립하기보다 `ArrayDatasetBuilder`를 쓰는 것이 권장된다.

예:

```python
from array_binary_shard import (
    ArrayDatasetBuilder,
    LogicalType,
    PointColumnSpec,
    StorageType,
)

builder = ArrayDatasetBuilder(
    out_dir=".../array_binary_shards",
    sample_meta_path=".../sample_meta.parquet",
    point_schema=[
        PointColumnSpec(
            name="phase",
            storage_type=StorageType.INT32,
            logical_type=LogicalType.INTEGER,
        ),
        PointColumnSpec(
            name="state_code",
            storage_type=StorageType.UINT32,
            logical_type=LogicalType.CATEGORICAL,
        ),
    ],
)

builder.add_trace(
    sample_id=0,
    feature_key="feature_a",
    columns={
        "phase": [10, 11, 12],
        "state_code": ["OK", "OK", "WARN"],
    },
)

manifest_path = builder.build_shards()
```

### 5.2 bundle stage

현재 구현은 내부적으로 바로 shard에 쓰지 않고, 먼저 sample-major bundle stage를 만든 뒤 shard로 변환한다.

명시적으로 중간 산출물을 보고 싶으면:

- `finish_bundles()`
- `build_shards(cleanup_bundles=...)`

를 쓴다.

예:

```python
bundle_manifest_path = builder.finish_bundles()
manifest_path = builder.build_shards(cleanup_bundles=False)
```

### 5.3 metadata helper

sample / feature metadata는 helper 함수로 만들 수 있다.

- `write_sample_meta(records, out_path)`
- `write_feature_meta(records, out_path)`

입력은 `list[dict]`이고, dense id는 row order 기준으로 자동 생성되거나 검증된다.

### 5.4 discovered-feature mode

feature 목록을 처음부터 모를 때는 discovered-feature mode를 쓸 수 있다.

- `feature_key`를 처음 본 순서대로 dense `feature_id`를 할당
- 기본적으로 생성되는 `feature_meta.parquet`는 `feature_id`, `feature_key`만 가진다

추가 feature metadata가 필요하면 bundle stage 이후에:

- `update_feature_meta(records, on=..., require_all=...)`

로 column을 merge할 수 있다.

---

## 6. Manifest

최상위 manifest 예시는 대략 아래와 같다.

```json
{
  "format": "array-binary-shard",
  "version": 3,
  "endianness": "little",
  "sample_meta_path": "sample_meta.parquet",
  "feature_meta_path": "feature_meta.parquet",
  "n_samples": 2000,
  "n_features": 1024,
  "shard_path": "array_binary_feature_shards",
  "n_shards": 8,
  "samples_per_block": 16,
  "blocks_per_feature": 125,
  "feature_id_dtype": "INT32",
  "flags_dtype": "UINT8",
  "offset_dtype": "INT64",
  "default_codec": "none",
  "id_scheme": "dense_row_ids",
  "sample_key_col": "sample_key",
  "feature_key_col": "feature_key",
  "point_schema": [
    {
      "name": "phase",
      "storage_type": "int32",
      "logical_type": "integer"
    },
    {
      "name": "state_code",
      "storage_type": "uint32",
      "logical_type": "categorical",
      "dictionary_path": "categorical_dictionaries/state_code.json"
    },
    {
      "name": "event_type",
      "storage_type": "uint32",
      "logical_type": "categorical",
      "dictionary_path": "categorical_dictionaries/event_type.json"
    }
  ],
  "shards": [
    {
      "shard_id": 0,
      "feature_id_start": 0,
      "feature_id_end": 127,
      "feature_count": 128,
      "block_count": 16000,
      "blocks_index_name": "shard_0000.blocks.idx",
      "blocks_data_name": "shard_0000.blocks.bin"
    }
  ]
}
```

주의:

- v3 manifest는 `point_schema`를 반드시 포함한다.
- `time_dtype`, `value_dtype` 같은 `time/value` 전용 필드는 v3 manifest에 쓰지 않는다.

---

## 7. `blocks.idx`

### 7.1 파일 전체 구조

```text
[64-byte file header][record 0][record 1][record 2]...
```

### 7.2 file header

`blocks.idx`와 `blocks.bin`은 같은 공통 64-byte file header 구조를 사용한다.

Python struct:

```python
struct.Struct("<8sHHHHQQI28x")
```

필드 의미:

| offset | size | type | meaning |
|---|---:|---|---|
| 0 | 8 | bytes | magic |
| 8 | 2 | uint16 | version |
| 10 | 2 | uint16 | header_bytes |
| 12 | 2 | uint16 | record_bytes |
| 14 | 2 | uint16 | flags |
| 16 | 8 | uint64 | entry_count |
| 24 | 8 | uint64 | aux_count |
| 32 | 4 | uint32 | shard_id |
| 36 | 28 | reserved | future expansion |

`blocks.idx`에서:

- `magic = b"ABLOCKIX"`
- `version = 3`
- `header_bytes = 64`
- `record_bytes = 32`
- `entry_count = block_count`
- `aux_count = feature_count`
- `shard_id = shard_id`

`reserved`는 현재 항상 0이고 reader는 무시한다. 미래 확장용이다.

### 7.3 record 구조

`blocks.idx`의 record는 32 bytes다.

NumPy dtype:

```python
[
  ("data_offset", "<u8"),
  ("data_length", "<u8"),
  ("point_count", "<u8"),
  ("codec", "u1"),
  ("block_flags", "u1"),
  ("reserved0", "<u2"),
  ("crc32_optional", "<u4"),
]
```

필드 의미:

| offset | size | type | meaning |
|---|---:|---|---|
| 0 | 8 | uint64 | `blocks.bin` 안 payload의 시작 offset |
| 8 | 8 | uint64 | payload byte length |
| 16 | 8 | uint64 | 이 block 안 전체 point 수 |
| 24 | 1 | uint8 | codec id (`0=none`, `1=zstd`) |
| 25 | 1 | uint8 | block flags, 현재는 0 |
| 26 | 2 | uint16 | reserved |
| 28 | 4 | uint32 | optional checksum, 현재는 0 |

### 7.4 record index 계산

한 shard 안에서 `blocks.idx`는 **feature x block** 2차원 grid를 1차원 배열로 펼친 구조다.

정의:

- `local_feature = feature_id - shard.feature_id_start`
- `block_id = sample_id // samples_per_block`
- `record_index = local_feature * blocks_per_feature + block_id`

즉:

- 같은 feature의 block들은 `blocks_per_feature` 개씩 연속 배치
- feature가 바뀌면 그 다음 묶음으로 넘어감

예:

- `feature_id_start = 128`
- `feature_id = 140`
- `local_feature = 12`
- `blocks_per_feature = 125`
- `block_id = 7`

이면:

- `record_index = 12 * 125 + 7 = 1507`

이고, `blocks.idx[1507]`가 찾고 싶은 block record다.

---

## 8. `blocks.bin`

### 8.1 파일 전체 구조

```text
[64-byte file header][payload 0][payload 1][payload 2]...
```

### 8.2 file header

`blocks.bin`도 같은 64-byte file header를 쓴다.

차이:

- `magic = b"ABLOCKSB"`
- `record_bytes = 0`
- `entry_count = block_count`
- `aux_count = total data bytes after the 64-byte header`

즉 `blocks.bin`은 fixed-size record array가 아니라 variable-length payload들의 연속 저장소다.

### 8.3 payload header

각 payload는 먼저 48-byte payload header를 가진다.

Python struct:

```python
struct.Struct("<iiqIBBHQIIII")
```

필드 의미:

| offset | size | type | meaning |
|---|---:|---|---|
| 0 | 4 | int32 | `feature_id` |
| 4 | 4 | int32 | `block_id` |
| 8 | 8 | int64 | `sample_id_start` |
| 16 | 4 | uint32 | `sample_count` |
| 20 | 1 | uint8 | `codec` |
| 21 | 1 | uint8 | `header_flags` |
| 22 | 2 | uint16 | `schema_column_count` |
| 24 | 8 | uint64 | `point_count` |
| 32 | 4 | uint32 | `flags_bytes` |
| 36 | 4 | uint32 | `offsets_bytes` |
| 40 | 4 | uint32 | `encoded_columns_payload_bytes` |
| 44 | 4 | uint32 | reserved |

주의:

- 코드와 payload header에서도 `sample_id_start`를 사용한다.
- v3 의미상 이것은 **그 block의 첫 dense `sample_id`**다.
- 항상 `block_id * samples_per_block`와 같다.

### 8.4 payload body

payload body는 아래 순서다.

```text
[sample_flags][sample_offsets][encoded_columns_payload]
```

`encoded_columns_payload`를 codec으로 decode하면:

```text
[col0][col1][col2]...[colN]
```

가 되고, 이때 column 순서는 `manifest.point_schema` 순서와 정확히 일치한다.

각 column byte length는:

```text
point_count * itemsize(storage_type)
```

로 계산한다.

예를 들어 schema가:

- `phase: int32`
- `state_code: uint32`
- `event_type: uint32`

이면 column payload는:

```text
[phase int32[point_count]]
[state_code uint32[point_count]]
[event_type uint32[point_count]]
```

순서로 들어간다.

---

## 9. `sample_flags` 와 `sample_offsets`

### 9.1 `sample_offsets`

길이:

- `sample_count + 1`

dtype:

- `int64`

의미:

- block 안의 각 sample trace가 point payload 안에서 어디서 시작하고 끝나는지 알려준다

예:

```text
sample_offsets = [0, 3, 7, 7, 10]
```

이면:

- sample 0 -> points `[0:3]`
- sample 1 -> points `[3:7]`
- sample 2 -> points `[7:7]`   (empty)
- sample 3 -> points `[7:10]`

### 9.2 `sample_flags`

길이:

- `sample_count`

dtype:

- `uint8`

현재 사용 bit:

| bit | hex | meaning |
|---|---|---|
| 0 | `0x01` | present |
| 1 | `0x02` | empty |
| 2..7 | reserved | currently 0 |

주의:

- 현재 구현은 `present`, `empty`만 사용한다.
- 나머지 비트는 예약 상태이며 reader는 무시한다.

---

## 10. End-to-End 조회 예제

가정:

- `samples_per_block = 16`
- `blocks_per_feature = 125`
- 요청:
  - `feature_id = 140`
  - `sample_id = 118`

### 10.1 shard 선택

manifest의 shard 목록에서:

- `feature_id_start <= 140 <= feature_id_end`

를 만족하는 shard를 찾는다.

예를 들어:

- `feature_id_start = 128`
- `feature_id_end = 255`

인 shard 1이 선택된다.

### 10.2 block 계산

- `block_id = 118 // 16 = 7`
- `relative_sample = 118 % 16 = 6`

### 10.3 `record_index` 계산

- `local_feature = 140 - 128 = 12`
- `record_index = 12 * 125 + 7 = 1507`

### 10.4 `blocks.idx` 읽기

`blocks.idx[1507]`에서:

- `data_offset`
- `data_length`
- `point_count`
- `codec`

를 읽는다.

### 10.5 `blocks.bin` payload 읽기

`blocks.bin`에서:

- `seek(data_offset)`
- `read(data_length)`

후 payload header를 파싱한다.

여기서:

- `sample_id_start == 112`
- `sample_count == 16`

이어야 한다.

### 10.6 sample slice 계산

`sample_offsets`가 예를 들어:

```text
[0, 3, 5, 8, 8, 11, 14, 20, ...]
```

이면 `relative_sample = 6`인 sample의 구간은:

- `start = sample_offsets[6]`
- `end = sample_offsets[7]`

이다.

각 column array에서 `[start:end]`를 slice 하면 그 sample trace가 된다.

### 10.7 key 기반 조회

만약 요청이:

- `feature_key = "feature_000140"`
- `sample_key = "sample_000118"`

이면 먼저 metadata에서:

- `feature_key -> feature_id`
- `sample_key -> sample_id`

를 resolve한 뒤, 이후 경로는 위 dense-id fast path와 동일하다.

---

## 11. 서버 API

현재 FastAPI 구현 기준으로 array 관련 주요 엔드포인트는 다음과 같다.

- `POST /array-schema`
  - `point_schema` 조회
  - `include_dictionaries=true`면 categorical dictionary도 함께 반환
- `POST /array-feature`
  - trace 조회
  - `decode_categorical=true`면 categorical code를 label로 decode
  - `temporal_format="iso" | "raw_ns"`로 temporal column 직렬화 형식을 선택

`/array-feature` 응답은 더 이상 `time` / `value`만 특별 취급하지 않고:

```json
{
  "sample_id": 0,
  "sample_key": "sample_000000",
  "flags": 1,
  "columns": {
    "phase": [10, 11, 12],
    "state_code": ["OK", "OK", "WARN"]
  }
}
```

같이 generic `columns` 맵으로 내려간다.

---

## 12. Validation Checklist

독립 reader를 구현할 때 최소한 아래 검증은 하는 것이 좋다.

### 12.1 manifest

- `format == "array-binary-shard"`
- `version == 3`
- `endianness == "little"`
- `id_scheme == "dense_row_ids"`
- `samples_per_block > 0`
- `blocks_per_feature == ceil(n_samples / samples_per_block)`

### 12.2 metadata

- `sample_meta.parquet` row count == `n_samples`
- `feature_meta.parquet` row count == `n_features`
- `sample_id` column이 있으면 row order와 정확히 일치
- `feature_id` column이 있으면 row order와 정확히 일치
- `sample_key`, `feature_key`가 있으면 unique / non-null

### 12.3 `blocks.idx`

- magic / version / header size 검증
- `record_bytes == 32`
- `entry_count == feature_count * blocks_per_feature`

### 12.4 `blocks.bin`

- magic / version / header size 검증
- payload header의 `feature_id`, `block_id`가 기대값과 일치
- payload header의 `sample_id_start`, `sample_count`가 계산값과 일치
- `schema_column_count == len(point_schema)`

### 12.5 payload

- `sample_flags` 길이 == `sample_count`
- `sample_offsets` 길이 == `sample_count + 1`
- decode한 column payload 총 byte 수 == `sum(point_count * itemsize)`
- 마지막 column까지 읽은 뒤 trailing byte가 남지 않아야 함

### 12.6 reserved field

- `reserved`, `reserved0`, checksum 필드는 현재 0으로 쓰는 것이 권장
- reader는 현재는 무시해도 됨
- 미래 버전에서 의미가 생길 수 있으므로, 값이 0이 아니어도 당장 실패시키지 않는 정책도 가능

---

## 13. 요약

v3를 한 문장으로 정리하면:

**dense id 기반 fast lookup 구조를 유지하면서, `time` / `value` 특수 포맷을 manifest-defined point-column schema로 일반화한 binary shard format**이다.

핵심 요약:

- `sample_id`, `feature_id`는 dense integer id
- `sample_key`, `feature_key`는 optional stable external key
- `time`, `value`는 더 이상 필수 아님
- `point_schema`는 manifest마다 고정
- categorical은 `uint32` code + dictionary JSON
- temporal은 `int64` + `timestamp_ns` / `timedelta_ns`
- payload는 `[flags][offsets][encoded_columns_payload]`
- 빠른 조회는 `record_index` 계산 + `data_offset` seek로 유지
