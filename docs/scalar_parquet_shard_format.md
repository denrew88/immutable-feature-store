# Scalar Parquet Shard Format

이 문서는 현재 repo에서 사용하는 **scalar feature 전용 shard format**을 설명한다.

중요한 전제:

- 현재 scalar는 custom binary format이 아니라 **Parquet + sidecar metadata** 구조다.
- 내부 id는 dense integer id다.
  - `sample_id == sample_meta.parquet`의 row index
  - `feature_id == feature_meta.parquet`의 row index
- shard artifact는 standalone folder로 묶인다.
- selection fast-path를 위해 `selection_stats/` sidecar를 별도로 둔다.

즉 scalar는 array binary v3처럼 payload를 완전히 바이너리로 직접 짜지는 않았지만,
운영 관점에서는 이미 **standalone immutable artifact**처럼 다룰 수 있게 정리된 상태다.

---

## 1. 전체 구조

scalar shard dataset은 보통 아래처럼 생긴다.

```text
scalar_shard_dataset/
  shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  selection_stats/
    y.parquet
    y_alt.parquet
    ...
  feature_shards/
    shard_0000.parquet
    shard_0001.parquet
    ...
```

각 파일의 역할:

- `shard_manifest.json`
  - dataset 전체 메타데이터
  - dense id 규칙, shard 개수, locator 경로, selection stats 경로를 정의
- `sample_meta.parquet`
  - sample row order를 정의
  - **row index가 곧 `sample_id`**
- `feature_meta.parquet`
  - feature row order를 정의
  - **row index가 곧 `feature_id`**
- `feature_locator.parquet`
  - `feature_id -> (shard_id, offset_in_shard)` 매핑
- `selection_stats/<y>.parquet`
  - 특정 target column에 대한 precomputed candidate stats
- `feature_shards/shard_XXXX.parquet`
  - 실제 scalar feature row 저장

중요:

- manifest 안 경로는 **manifest 기준 relative path**로 저장된다.
- 따라서 폴더를 통째로 옮겨도 그대로 읽을 수 있다.

---

## 2. Dense Id 와 External Key

### 2.1 dense id

현재 scalar 규칙:

- `sample_id == sample_meta.parquet`의 row index
- `feature_id == feature_meta.parquet`의 row index

즉:

- `sample_meta.parquet`의 0번째 row -> `sample_id = 0`
- `feature_meta.parquet`의 17번째 row -> `feature_id = 17`

코드는 metadata를 읽을 때 이 dense 규칙을 검증한다.

### 2.2 external key

외부 시스템과 연동하려면 metadata에 stable key를 둔다.

- `sample_key`
- `feature_key`

서버 API는 id와 key를 모두 지원한다.

- scalar 조회:
  - `feature_id` 또는 `feature_key`
  - `sample_ids` 또는 `sample_keys`
- selection 응답:
  - `selected_feature_ids`
  - `selected_feature_keys`

권장 규칙:

- `sample_key`는 non-null, unique
- `feature_key`는 non-null, unique

---

## 3. `sample_meta.parquet`

### 3.1 역할

`sample_meta.parquet`는 dense sample id를 정의하는 authoritative metadata다.

row order가 곧 `sample_id`다.

### 3.2 현재 코드가 기대하는 최소 컬럼

scalar build / selection 기준 최소 필요 컬럼:

- `y` 또는 선택 가능한 target column
- `sample_path`

실무적으로는 보통 아래도 같이 둔다.

- `sample_id`
- `sample_key`

### 3.3 `sample_id`

`sample_id` 컬럼이 존재하면:

- 반드시 `0..n_samples-1`
- 그리고 row order와 정확히 일치

해야 한다.

즉 아래는 허용되지 않는다.

- sparse sample id
- offset된 sample id
- row order와 다른 sample id

### 3.4 `sample_path`

`sample_path`는 per-sample long-format parquet 파일 경로다.

각 sample parquet는 보통 아래 컬럼을 가진다.

- `feature_id`
- `value`

즉 scalar builder는 sample-major long-format 입력을 읽어서
feature-major shard로 재배치한다.

### 3.5 자유 컬럼

그 외 컬럼은 자유롭게 둘 수 있다.

예:

- `split`
- `patient_id`
- `visit_id`
- `y_alt`
- `group`

selection에서 어떤 target을 쓸지는 요청 시 `y_col`로 고를 수 있다.

---

## 4. `feature_meta.parquet`

### 4.1 역할

`feature_meta.parquet`는 dense feature id를 정의하는 authoritative metadata다.

row order가 곧 `feature_id`다.

### 4.2 현재 코드가 기대하는 핵심 컬럼

권장 핵심 컬럼:

- `feature_id`
- `feature_key`

`feature_id`가 있으면 row order와 정확히 일치해야 한다.

### 4.3 자유 컬럼

그 외 metadata column은 자유롭게 둘 수 있다.

예:

- `group`
- `display_name`
- `modality`
- `feature_kind`

selection 결과는 내부적으로 dense `feature_id`를 사용하지만, 최종 응답에는 `feature_key`도 함께 실어준다.

---

## 5. 최종 shard row schema

각 scalar shard parquet의 row 하나는 feature 하나를 나타낸다.

현재 row schema:

- `feature_id: Int32`
- `value_len: Int32`
- `values_blob: Binary`
- `valid_blob: Binary`

즉 feature 하나에 대해:

- length `n_samples`의 float64 value vector
- length `n_samples`의 uint8 validity vector

를 blob으로 저장한다.

### 5.1 `value_len`

현재 dense scalar format에서는 보통:

- `value_len == n_samples`

이다.

즉 row마다 vector 길이가 dataset-wide sample count와 같다.

### 5.2 `values_blob`

의미:

- little-endian float64 buffer

길이:

- `value_len * 8 bytes`

decode:

```python
np.frombuffer(values_blob, dtype="<f8", count=value_len)
```

### 5.3 `valid_blob`

의미:

- uint8 mask buffer

길이:

- `value_len bytes`

decode:

```python
np.frombuffer(valid_blob, dtype=np.uint8, count=value_len)
```

관례:

- `1` = present
- `0` = missing

### 5.4 왜 blob 형태를 쓰나

scalar는 point-level variable trace가 아니므로 array처럼 block payload 구조까지 필요하지 않다.

대신 feature-major row 하나에:

- value vector 전체
- valid vector 전체

를 저장하면 lookup과 selection 구현이 단순해진다.

---

## 6. `feature_locator.parquet`

### 6.1 역할

`feature_locator.parquet`는 dense `feature_id`를 실제 shard 위치로 바꾸는 sidecar다.

현재 기본 컬럼:

- `feature_id: Int32`
- `global_rank: Int32`
- `shard_id: Int32`
- `offset_in_shard: Int32`

### 6.2 의미

- `feature_id`
  - dense feature id
- `global_rank`
  - 현재는 dense feature ordering을 다시 나타내는 용도
- `shard_id`
  - feature가 들어 있는 shard 번호
- `offset_in_shard`
  - 해당 shard parquet 안에서 몇 번째 row인지

### 6.3 lookup 흐름

`feature_id = 123`을 조회하려면:

1. locator에서 `feature_id = 123` row 찾기
2. `shard_id`, `offset_in_shard` 읽기
3. `feature_shards/shard_XXXX.parquet` 열기
4. `offset_in_shard` 번째 row 읽기
5. `values_blob`, `valid_blob` decode

즉 array처럼 block index를 계산하는 구조는 아니고, locator가 직접 row 위치를 들고 있다.

### 6.4 legacy note

예전에는 locator에 `r2y`, `n_y_overlap` 같은 candidate stats를 같이 넣던 구조가 있었다.

현재는 그 역할을 `selection_stats/*.parquet`가 맡고, locator는 위치 정보만 갖는 것이 기본이다.

코드는 old locator도 읽을 수 있게 일부 호환 경로를 남겨두고 있다.

---

## 7. `selection_stats/`

### 7.1 왜 별도 sidecar가 필요한가

selection을 빠르게 하려면 target column `y_col`마다
feature-vs-y candidate stats를 미리 계산해두는 것이 유리하다.

그래서 현재 scalar artifact는 `selection_stats/`를 별도 폴더로 둔다.

예:

```text
selection_stats/
  y.parquet
  y_alt.parquet
```

### 7.2 파일 하나의 row schema

현재 stats parquet는 feature 하나당 row 하나다.

컬럼:

- `feature_id: Int32`
- `shard_id: Int32`
- `offset_in_shard: Int32`
- `r2y: Float64`
- `n_y_overlap: Int32`

즉 특정 `y_col`에 대해:

- feature가 target과 얼마나 상관이 높은지
- 유효한 pair 수가 몇 개인지

를 빠르게 읽을 수 있다.

### 7.3 manifest와의 연결

manifest는 `selection_stats` 매핑을 가진다.

예:

```json
"selection_stats": {
  "y": "selection_stats/y.parquet",
  "y_alt": "selection_stats/y_alt.parquet"
}
```

selection 요청 시:

- `y_col`이 이 매핑에 있으면 fast-path
- 없으면 shard를 다시 읽는 fallback path

로 동작한다.

---

## 8. `shard_manifest.json`

현재 manifest 예시는 대략 아래와 같다.

```json
{
  "sample_meta_path": "sample_meta.parquet",
  "feature_meta_path": "feature_meta.parquet",
  "n_samples": 2000,
  "n_features": 50000,
  "shard_path": "feature_shards",
  "n_shards": 8,
  "feature_locator_path": "feature_locator.parquet",
  "feature_locator_format": "parquet_v1",
  "feature_id_dtype": "INT32",
  "values_dtype": "blob_float64_le_len",
  "valid_dtype": "blob_uint8_len",
  "id_scheme": "dense_row_ids",
  "sample_key_col": "sample_key",
  "feature_key_col": "feature_key",
  "target_shard_bytes": 33554432,
  "selection_stats": {
    "y": "selection_stats/y.parquet",
    "y_alt": "selection_stats/y_alt.parquet"
  }
}
```

핵심 필드:

- `sample_meta_path`, `feature_meta_path`
  - standalone metadata 경로
- `n_samples`, `n_features`
  - dense sample / feature count
- `shard_path`
  - `feature_shards/` 폴더
- `feature_locator_path`
  - locator parquet
- `values_dtype`, `valid_dtype`
  - blob encoding 의미
- `id_scheme`
  - 현재는 `dense_row_ids`
- `selection_stats`
  - precomputed target stats sidecar 매핑

### 8.1 legacy field

`stats_y_col`은 예전 single-target 설계 흔적으로 dataclass에 남아 있을 수 있지만,
현재 권장 구조는 `selection_stats` 매핑이다.

즉 새 artifact를 설명할 때는 `selection_stats`를 기준으로 보면 된다.

---

## 9. Build 흐름

현재 Python scalar build는 `build_shards_from_sample_major(...)`가 담당한다.

입력:

- `sample_meta.parquet`
- `feature_meta.parquet`
- sample-major per-sample parquet files

흐름:

1. `sample_meta.parquet` 읽기
2. `feature_meta.parquet` 읽기
3. dense id와 key uniqueness 검증
4. target shard size 기준으로 feature range를 shard에 배분
5. `_tmp/` 아래에 shard별 memmap backing file 생성
   - `shard_XXXX_values.dat`
   - `shard_XXXX_valid.dat`
6. sample-major 파일을 한 번씩 읽으며 memmap에 직접 채움
7. selection stats를 계산
8. 각 shard parquet row를 materialize
9. locator와 selection stats sidecar 작성
10. manifest 작성
11. backing file cleanup, 빈 `_tmp` 삭제

즉 array처럼 append-only spill은 아니고,
scalar는 **shard별 dense memmap backing file** 전략을 쓴다.

### 9.1 direct-ingestion builder

현재는 sample-major parquet를 사용자가 직접 조립하지 않도록
`ScalarDatasetBuilder`를 사용할 수 있다.

권장 public API:

- `write_sample(sample_id, values=...)`
- `with builder.open_sample(sample_id) as sample: ...`
- `finish_sample_major()`
- `update_feature_meta(...)`
- `build_shards(keep_sample_major=...)`

예:

```python
from fs.config import ScalarShardBuildOptions
from fs.scalar import ScalarDatasetBuilder

builder = ScalarDatasetBuilder(
    out_dir=".../scalar_shards",
    sample_meta_path=".../sample_meta.parquet",
    build_options=ScalarShardBuildOptions(
        target_shard_mb=32,
        stats_y_cols=("y", "y_alt"),
    ),
)

builder.write_sample(0, {"feature_a": 1.23, "feature_b": 4.56})
builder.write_sample(1, {"feature_a": 7.89})
manifest_path = builder.build_shards()
```

### 9.2 왜 sample 단위 API를 쓰는가

scalar builder는 array와 달리 **sample 단위로 즉시 flush**하는 것이 자연스럽다.

이유:

- intermediate format 자체가 sample-major 파일 모음이다
- sample 하나의 feature map만 메모리에 들고 있으면 된다
- 사용자가 sample을 재방문하는 실수를 쉽게 막을 수 있다

현재 builder 규칙:

- sample은 어떤 순서로 추가해도 된다
- 같은 `sample_id`를 두 번 쓰면 에러
- `write_sample(...)`는 호출 직후 그 sample parquet를 바로 쓴다
- `with builder.open_sample(...)`는 context 종료 시 flush한다

즉 메모리 사용량은 기본적으로 **현재 sample 하나의 feature map** 크기에 비례한다.

### 9.3 visible sample-major stage

builder는 intermediate sample-major stage를 명시적으로 노출한다.

기본 위치 예:

```text
<out_dir>/
  sample_major_stage/
    sample_meta.parquet
    feature_meta.parquet
    samples/
      sample_000000.parquet
      sample_000001.parquet
      ...
```

`finish_sample_major()`를 호출하면 이 중간 산출물을 확정한다.

이후:

- `update_feature_meta(...)`로 discovered-feature mode의 extra column을 merge 가능
- `build_shards()`로 최종 standalone scalar shard artifact를 생성

### 9.4 cleanup 주의점

`build_shards(keep_sample_major=False)`가 기본이고, 이 경우 intermediate sample-major stage를 삭제한다.

이 경우:

- serving
- precomputed `selection_stats`를 사용하는 selection fast-path

는 계속 동작한다.

하지만:

- manifest에 없는 새로운 `y_col`에 대해
- sample-major 파일을 다시 읽어 candidate를 계산하는 fallback selection

입력은 사라진다.

반대로 `build_shards(keep_sample_major=True)`를 쓰면 intermediate sample-major stage를 유지한다.

즉 기본 설정은 artifact를 더 깔끔하게 만들고, `keep_sample_major=True`는
fallback selection 입력까지 남겨두는 선택이라고 이해하면 된다.

---

## 10. End-to-End 조회 예제

가정:

- 요청:
  - `feature_id = 123`
  - `sample_ids = [0, 3, 7]`

### 10.1 locator 조회

`feature_locator.parquet`에서 `feature_id = 123` row를 찾는다.

예:

- `shard_id = 2`
- `offset_in_shard = 411`

### 10.2 shard row 읽기

`feature_shards/shard_0002.parquet`에서 411번째 row를 읽는다.

거기서:

- `value_len = 2000`
- `values_blob`
- `valid_blob`

를 얻는다.

### 10.3 blob decode

- `values = float64[2000]`
- `valid = uint8[2000]`

를 만든다.

### 10.4 sample slice

요청한 sample ids가 `[0, 3, 7]`이면:

- `values[0]`, `valid[0]`
- `values[3]`, `valid[3]`
- `values[7]`, `valid[7]`

을 응답에 넣는다.

### 10.5 key 기반 조회

만약 요청이:

- `feature_key = "feature_000123"`
- `sample_keys = ["sample_000000", "sample_000003"]`

이면 먼저 metadata에서:

- `feature_key -> feature_id`
- `sample_key -> sample_id`

를 resolve한 뒤, 이후 경로는 dense-id fast path와 동일하다.

---

## 11. Selection 흐름

### 11.1 fast-path

`/run-selection` 또는 `scripts.run_selection`에서:

- 요청한 `y_col`
- manifest `selection_stats`

를 확인한다.

해당 `y_col` stats가 있으면:

1. `selection_stats/<y_col>.parquet` 읽기
2. `r2y`, `n_y_overlap`로 candidate filtering
3. `ParquetShardReader`로 필요한 feature row만 읽으며 incremental selection

즉 candidate 단계가 빠르다.

### 11.2 fallback path

요청한 `y_col`에 대한 precomputed stats가 없으면:

1. `sample_meta.parquet`에서 `y_col` 읽기
2. shard parquet 전체를 batch scan
3. `r2y` candidate 계산
4. 이후 incremental selection 수행

즉 동작은 가능하지만 더 느리다.

---

## 12. 서버 API

현재 FastAPI 기준 scalar 관련 주요 엔드포인트는:

- `POST /scalar-feature`
- `POST /run-selection`
- `GET /healthz`
- `GET /cache-stats`

### 12.1 `/scalar-feature`

입력:

- `manifest_path`
- `feature_id` 또는 `feature_key`
- `sample_ids` 또는 `sample_keys`
- `sanitize_nonfinite`

응답:

```json
{
  "manifest_path": "...",
  "feature_id": 123,
  "feature_key": "feature_000123",
  "sample_count": 2,
  "values": [
    {
      "sample_id": 1,
      "sample_key": "sample_000001",
      "present": true,
      "value": 0.123
    },
    {
      "sample_id": 7,
      "sample_key": "sample_000007",
      "present": false,
      "value": null
    }
  ]
}
```

`sample_row`는 응답에 더 이상 포함하지 않는다.

### 12.2 `/run-selection`

입력:

- `manifest_path`
- `y_col`
- threshold / cap / batch 관련 selection 옵션

응답 핵심:

- `selected_feature_ids`
- `selected_feature_keys`
- `candidate_count`
- `selected_count`
- `used_locator_stats`
- `candidate_build_ms`
- `selection_ms`
- `elapsed_ms`

---

## 13. Validation Checklist

독립 구현이나 운영 검증 시 최소한 아래는 확인하는 것이 좋다.

### 13.1 metadata

- `sample_meta.parquet` row count == `n_samples`
- `feature_meta.parquet` row count == `n_features`
- `sample_id`가 있으면 row order와 일치
- `feature_id`가 있으면 row order와 일치
- `sample_key`, `feature_key`는 unique / non-null

### 13.2 locator

- `feature_locator.parquet` row count == `n_features`
- `feature_id`는 dense `0..n_features-1`
- `shard_id`, `offset_in_shard`가 실제 shard row 범위를 벗어나지 않아야 함

### 13.3 shard row

- `feature_id`는 locator와 일치
- `value_len == n_samples`
- `len(values_blob) == value_len * 8`
- `len(valid_blob) == value_len`

### 13.4 selection stats

- stats 파일 row count == `n_features`
- `feature_id`, `shard_id`, `offset_in_shard`가 locator와 일치
- `n_y_overlap >= -1`
- `r2y`는 `NaN` 허용

---

## 14. 요약

현재 scalar 포맷을 한 문장으로 정리하면:

**dense id 기반 metadata + locator parquet + feature-major shard parquet + optional selection stats sidecar로 구성된 standalone scalar artifact**다.

핵심 요약:

- `sample_id`, `feature_id`는 dense row id
- `sample_key`, `feature_key`는 optional external key
- 실제 shard row는 `values_blob`, `valid_blob`를 가진 feature-major parquet row
- `feature_locator.parquet`가 lookup 위치를 알려줌
- `selection_stats/<y>.parquet`가 candidate fast-path를 제공
- artifact 전체는 한 폴더로 묶여 standalone하게 이동 가능
