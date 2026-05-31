# Scalar Parquet Shard Format

## 현재 구현 요약

scalar에는 현재 세 가지 관련 포맷이 있습니다.

1. `raw sample stage`
   - `ScalarRawDatasetBuilder`가 사용하는 재개 가능 ingest 포맷입니다.
   - sample 하나가 parquet 파일 하나입니다.
   - sample은 임의 순서로 만들 수 있습니다.
   - 완료된 sample은 `raw_samples.jsonl`에 기록됩니다.
   - shard 생성은 모든 raw sample 작성이 끝난 뒤 별도로 수행합니다.

2. `blob shard`
   - 기존 최종 serving 포맷입니다.
   - feature 하나가 parquet row 하나입니다.
   - 모든 sample 값은 `values_blob`, `valid_blob`에 dense vector로 들어갑니다.
   - feature batch 조회와 파일 크기에 유리합니다.

3. `dense-long shard`
   - 새로 추가된 최종 parquet 포맷입니다.
   - 모든 `(feature_id, sample_id)` 조합을 row로 저장합니다.
   - missing은 row 생략이 아니라 `mask=0`으로 표현합니다.
   - 물리 정렬은 `feature_id asc, sample_id asc`입니다.
   - 기본 row group은 feature 128개 단위입니다. feature-id filter는 해당 feature가 들어 있는 row group만 읽도록 pruning됩니다.

## Raw Sample Stage

`raw sample stage`는 array sample parquet의 raw-sample 방식과 같은 재개 모델입니다.

```text
scalar_raw_stage/
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  raw_samples/
    sample_000000000000.parquet
    sample_000000000017.parquet
```

각 raw sample parquet schema:

```text
sample_id   Int64
feature_id  Int32
value       Float64
```

특징:

- sample 파일은 `sample_id` 순서와 무관하게 생성할 수 있습니다.
- 같은 sample을 동시에 쓰지 않도록 sample별 `.lock` 파일을 사용합니다.
- 정상 commit은 `.tmp` 파일을 최종 `.parquet`로 rename한 뒤 `raw_samples.jsonl`에 기록합니다.
- `pending_sample_ids()`로 아직 작성되지 않은 sample 목록을 얻을 수 있습니다.
- `finish_stage()`는 raw 파일을 다시 쓰지 않고, raw sample parquet 경로들을 `sample_major_manifest.json`에 연결합니다.

## Dense-Long Shard

Dense-long shard artifact 구조:

```text
scalar_dense_long_shard/
  dense_long_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  selection_stats/
    y.parquet
  dense_long_parts/
    part_0000.parquet
    part_0001.parquet
```

`dense_long_parts/*.parquet` schema:

```text
feature_id  Int32
sample_id   Int64
mask        UInt8
value       Float64
```

의미:

- `mask=1`: `value`가 유효합니다.
- `mask=0`: missing입니다. 이때 `value`는 무시합니다.
- 모든 feature/sample 조합이 row로 존재하므로 sparse scatter로 missing 위치를 다시 채우지 않아도 됩니다.

Build 최적화:

- 입력 raw/bundle parquet를 파일 batch 단위로 한 번만 스캔합니다.
- feature-major memmap에 `values[feature, sample]`, `valid[feature, sample]`를 채웁니다.
- 같은 memmap에서 selection stats를 바로 계산합니다.
- 최종 parquet는 `feature_id, sample_id` 순서로 씁니다.
- 기본 row group size는 `n_samples * 128`입니다. 즉 feature 128개가 한 row group에 묶입니다. feature 하나만 읽을 때도 최대 128개 feature 묶음을 읽지만, row group 수와 metadata overhead가 줄어 파일 크기와 build 시간이 개선됩니다.

포맷 선택:

- feature batch 조회와 저장 크기가 중요하면 기존 `blob shard`가 유리합니다.
- 표준 parquet 디버깅, sample 기준 조회, row 기반 처리 호환성이 중요하면 `dense-long shard`가 유리합니다.
- sparse long 포맷은 현재 사용하지 않습니다.

## 조회 API Server

권장 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다. 이 서버는 기존 blob shard와 dense-long shard를 모두 열 수 있고, `scalar_format="auto"`일 때 manifest의 `format` 필드로 dense-long 여부를 자동 판별합니다.

실행:

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 endpoint:

- `POST /scalar/schema`: manifest 형식, sample/feature 수, key column, selection stats 목록을 반환합니다.
- `POST /scalar/features`: 여러 feature와 여러 sample의 값을 feature 중심 layout으로 반환합니다.
- `POST /scalar/sample`: sample 하나에 대한 feature vector를 반환합니다. dense-long shard에서는 sample 기준 parquet scan을 사용합니다.
- `POST /scalar/top-features`: `selection_stats/<y>.parquet`를 사용해 r2y 기준 상위 feature를 반환합니다.

요청 규칙:

- feature 축은 `feature_ids` 또는 `feature_keys` 중 하나만 받습니다.
- sample 축은 `sample_ids` 또는 `sample_keys` 중 하나만 받습니다.
- id와 key를 같은 축에 동시에 주면 400 에러입니다.
- 큰 요청은 `max_cells` 또는 `max_features`로 제한합니다.

예시:

```json
{
  "manifest_path": "data/scalar_dense_long/dense_long_shard_manifest.json",
  "feature_keys": ["feature_a", "feature_b"],
  "sample_keys": ["sample_000000", "sample_000001"]
}
```

이 문서는 현재 저장소에서 사용하는 scalar feature shard 형식을 설명한다.

중요한 특징:

- 최종 serving artifact는 Parquet 기반이다.
- `sample_id`, `feature_id`는 dense row id다.
- 전체 artifact는 standalone 폴더 하나로 이동할 수 있다.
- feature selection fast path를 위해 `selection_stats/` sidecar를 둔다.

## 1. 전체 구조

보통 scalar shard dataset은 아래처럼 생긴다.

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

각 파일 역할:

- `shard_manifest.json`
  - 전체 메타데이터
  - 모든 경로를 relative path로 저장
- `sample_meta.parquet`
  - sample 축 정의
- `feature_meta.parquet`
  - feature 축 정의
- `feature_locator.parquet`
  - `feature_id -> (shard_id, offset_in_shard)` 매핑
- `selection_stats/<y>.parquet`
  - 특정 target column에 대한 precomputed candidate stats
- `feature_shards/shard_XXXX.parquet`
  - 실제 scalar feature row 저장

## 2. Dense id 규칙

현재 scalar는 dense id를 사용한다.

- `sample_id == sample_meta.parquet`의 row index
- `feature_id == feature_meta.parquet`의 row index

즉:

- `sample_meta.parquet`의 0번째 row -> `sample_id = 0`
- `feature_meta.parquet`의 17번째 row -> `feature_id = 17`

외부 시스템과 연결할 때는 보통 다음 stable key를 쓴다.

- `sample_key`
- `feature_key`

## 3. `sample_meta.parquet`

`sample_meta.parquet`는 sample 축을 정의하는 authoritative metadata다.

권장 컬럼:

- `sample_id`
- `sample_key`
- `y`, `y_alt` 같은 target column

추가 컬럼은 자유롭게 넣을 수 있다.

예:

- `split`
- `patient_id`
- `visit_id`
- `group`

규칙:

- `sample_id`가 있으면 반드시 `0..n_samples-1` dense row order와 일치해야 한다.
- `sample_key`가 있으면 non-null, unique가 권장된다.

중요:

- 최종 shard artifact는 더 이상 `sample_path`에 의존하지 않는다.
- final lookup과 selection은 `sample_meta.parquet`, `feature_locator.parquet`, `feature_shards/`, `selection_stats/`만으로 동작한다.

## 4. `feature_meta.parquet`

`feature_meta.parquet`는 feature 축을 정의한다.

권장 컬럼:

- `feature_id`
- `feature_key`

추가 컬럼도 자유롭다.

예:

- `group`
- `display_name`
- `modality`
- `feature_kind`

규칙:

- `feature_id`가 있으면 반드시 `0..n_features-1` dense row order와 일치해야 한다.
- `feature_key`가 있으면 non-null, unique가 권장된다.

## 5. 최종 shard row schema

각 scalar shard parquet row는 feature 하나를 나타낸다.

현재 row schema:

- `feature_id: Int32`
- `value_len: Int32`
- `values_blob: Binary`
- `valid_blob: Binary`

### 5.1 `value_len`

현재 dense scalar format에서는 보통:

- `value_len == n_samples`

이다.

### 5.2 `values_blob`

물리 형식:

- little-endian float64 buffer

길이:

- `value_len * 8 bytes`

decode 예:

```python
np.frombuffer(values_blob, dtype="<f8", count=value_len)
```

### 5.3 `valid_blob`

물리 형식:

- uint8 mask buffer

길이:

- `value_len bytes`

의미:

- `1` = present
- `0` = missing

## 6. `feature_locator.parquet`

`feature_locator.parquet`는 serving 시점 lookup table이다.

현재 컬럼:

- `feature_id: Int32`
- `global_rank: Int32`
- `shard_id: Int32`
- `offset_in_shard: Int32`

의미:

- `feature_id`
  - dense feature id
- `global_rank`
  - dense feature 전체 순서
- `shard_id`
  - 해당 feature가 들어 있는 shard 번호
- `offset_in_shard`
  - shard parquet 안에서 몇 번째 row인지

lookup 흐름:

1. `feature_id = 123`인 locator row를 찾는다.
2. `shard_id`, `offset_in_shard`를 읽는다.
3. `feature_shards/shard_XXXX.parquet`를 연다.
4. `offset_in_shard` row를 읽는다.
5. `values_blob`, `valid_blob`를 decode한다.

예전에는 locator에 `r2y`, `n_y_overlap`가 같이 들어가던 구조도 있었지만, 현재는 그 역할을 `selection_stats/*.parquet`가 맡는다.

## 7. `selection_stats/`

selection fast path용 precomputed stats는 target column별 sidecar로 둔다.

예:

```text
selection_stats/
  y.parquet
  y_alt.parquet
```

각 파일 컬럼:

- `feature_id`
- `shard_id`
- `offset_in_shard`
- `r2y`
- `n_y_overlap`

manifest는 이를 다음처럼 연결한다.

```json
"selection_stats": {
  "y": "selection_stats/y.parquet",
  "y_alt": "selection_stats/y_alt.parquet"
}
```

## 8. `shard_manifest.json`

예:

```json
{
  "sample_meta_path": "sample_meta.parquet",
  "feature_meta_path": "feature_meta.parquet",
  "n_samples": 5000,
  "n_features": 20000,
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

중요 필드:

- `sample_meta_path`
- `feature_meta_path`
- `n_samples`, `n_features`
- `shard_path`
- `feature_locator_path`
- `id_scheme`
  - 현재는 `dense_row_ids`
- `selection_stats`

## 9. builder와 intermediate stage

### 9.1 direct-ingestion builder

현재는 사용자가 sample-major parquet를 직접 조립하지 않도록 `ScalarDatasetBuilder`를 제공한다.

### 9.2 visible sample-major stage

public builder는 intermediate stage를 명시적으로 노출한다.

Python:

- `finish_sample_major()`
- `build_shards(keep_sample_major=False)`

Java:

- `finishSampleMajor()`
- `buildShards(keepSampleMajor)`

### 9.3 intermediate 형식

현재 intermediate는 file-per-sample이 아니라 bundle 기반이다.

```text
sample_major_stage/
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  sample_bundles/
    bundle_000000.parquet
    bundle_000001.parquet
    ...
```

각 bundle row는 long-format scalar row다.

- `sample_id`
- `feature_id`
- `value`

### 9.4 cleanup

기본값:

- `keep_sample_major = false`

즉 최종 shard를 만든 뒤 intermediate stage는 지운다.

`keep_sample_major = true`로 두면 intermediate stage를 디버깅이나 fallback용으로 남길 수 있다.

## 10. selection 흐름

### 10.1 fast path

요청한 `y_col`이 `manifest.selection_stats`에 있으면:

1. `selection_stats/<y>.parquet`를 읽는다.
2. `r2y`, `n_y_overlap`로 candidate를 줄인다.
3. 줄어든 candidate에 대해서만 shard reader를 사용한다.

### 10.2 fallback

요청한 `y_col`이 `selection_stats`에 없으면:

1. `sample_meta.parquet`에서 target 값을 읽는다.
2. shard row를 다시 읽는다.
3. candidate stats를 on-the-fly로 계산한다.

이 fallback은 final artifact에 `sample_path`가 없어도 동작한다.

## 11. end-to-end 조회 예시

가정:

- `feature_id = 140`
- `sample_id = 118`

조회 순서:

1. `feature_locator.parquet`에서 `feature_id = 140` row를 찾는다.
2. 예를 들어:
   - `shard_id = 2`
   - `offset_in_shard = 17`
   를 얻는다.
3. `feature_shards/shard_0002.parquet`의 17번째 row를 읽는다.
4. `values_blob`, `valid_blob`를 decode한다.
5. sample position `118`을 본다.
6. `valid_blob[118] == 1`이면 `values_blob[118]`을 반환한다.
7. 아니면 missing이다.

key 기반 조회는 앞에 한 단계가 더 붙는다.

1. `feature_key -> feature_id`
2. `sample_key -> sample_id`
3. 이후는 같은 fast path

## 12. validation checklist

### metadata

- `sample_meta.parquet` row count == `n_samples`
- `feature_meta.parquet` row count == `n_features`
- `sample_id`, `feature_id`는 dense row order와 일치
- `sample_key`, `feature_key`는 가능하면 unique

### locator

- `feature_locator.parquet` row count == `n_features`
- `feature_id` unique
- `shard_id` 범위 유효
- `offset_in_shard` 범위 유효

### feature rows

- `value_len == n_samples`
- `values_blob.length == value_len * 8`
- `valid_blob.length == value_len`

### selection stats

- manifest에 적힌 각 stats 파일이 실제로 존재
- `feature_id`, `shard_id`, `offset_in_shard`가 locator와 일치

## 13. 요약

현재 scalar format은:

- Parquet 기반
- standalone
- dense-id 기반
- key 기반 조회 지원
- selection fast path 지원

최종 artifact는 `sample_path` 없이도 동작하고, public builder는 bundle 기반 intermediate stage를 거쳐 최종 standalone shard artifact를 만든다.
