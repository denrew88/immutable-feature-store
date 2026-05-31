# scalar-feature-shard

Dense-id 기반 scalar feature 데이터를 만들고 읽고 selection까지 수행하는 Python 패키지입니다.

## Public API

- `write_sample_meta(...)`, `write_feature_meta(...)`: dense row-id metadata를 작성합니다.
- `ScalarDatasetBuilder`: sample을 순차적으로 받아 bundle stage를 만들고 기존 blob shard를 생성합니다.
- `ScalarRawDatasetBuilder`: sample별 raw parquet를 독립적으로 만들고, 나중에 shard로 materialize합니다.
- `build_shard(...)`: 기존 sample-major parquet 또는 bundle manifest에서 blob shard를 생성합니다.
- `open_shard(...)`: 기존 blob shard reader입니다.
- `build_dense_long_shards_from_sample_bundles(...)`: bundle/raw sample rows에서 dense-long shard를 생성합니다.
- `open_dense_long_shard(...)`: dense-long shard reader입니다.
- `select_features(...)`, `run_selection(...)`: precomputed selection stats 또는 shard 본문을 사용해 feature selection을 수행합니다.

## Dense ID 규칙

- `sample_id == sample_meta.parquet`의 row index입니다.
- `feature_id == feature_meta.parquet`의 row index입니다.
- `sample_key`, `feature_key`는 외부 식별자이며, 내부 저장과 조회는 dense id를 기준으로 합니다.
- scalar missing value는 ingest 시 row를 쓰지 않는 방식으로 표현합니다.

## 두 가지 ingest 방식

### 순차 session builder

`ScalarDatasetBuilder`는 기존 방식입니다. `status().next_expected_sample_id`부터 순서대로 `write_sample(...)`을 호출해야 합니다. bundle checkpoint 단위로 resume할 수 있습니다.

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder

builder = ScalarDatasetBuilder.open_session(
    out_dir="data/scalar_blob",
    sample_meta_path="data/sample_meta.parquet",
    feature_meta_path="data/feature_meta.parquet",
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
)

st = builder.status()
for sample_id in range(st.next_expected_sample_id, n_samples):
    builder.write_sample(sample_id, {"feature_a": 1.23, "feature_b": 4.56})

manifest_path = builder.build_shards()
```

### Random raw-sample builder

`ScalarRawDatasetBuilder`는 array sample parquet의 raw-sample 방식과 같은 모델입니다. sample을 임의 순서로 만들 수 있고, 완료된 sample은 `raw_samples.jsonl`에 기록됩니다. 중단 후에는 `pending_sample_ids()`를 보고 남은 sample만 worker에 다시 배분하면 됩니다.

```python
from scalar_feature_shard import BuildOptions, ScalarRawDatasetBuilder

builder = ScalarRawDatasetBuilder.open_session(
    out_dir="data/scalar_raw",
    sample_meta_path="data/sample_meta.parquet",
    feature_meta_path="data/feature_meta.parquet",
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
)

for sample_id in [10, 3, 7, 1]:
    builder.write_sample(
        sample_id,
        {"feature_a": 1.23, "feature_b": None},
        skip_if_completed=True,
    )

print(builder.pending_sample_ids())
```

각 sample은 아래 파일 하나로 commit됩니다.

```text
raw_samples/
  sample_000000000003.parquet
  sample_000000000007.parquet
raw_samples.jsonl
raw_state.json
```

raw sample parquet schema는 다음과 같습니다.

```text
sample_id   Int64
feature_id  Int32
value       Float64
```

## Shard 포맷

### 기존 blob shard

기존 serving 포맷입니다. feature 하나가 parquet row 하나이고, 모든 sample 값은 binary blob으로 들어갑니다.

```text
feature_id   Int32
value_len    Int32
values_blob  Binary  # little-endian float64[n_samples]
valid_blob   Binary  # uint8[n_samples], 1=present, 0=missing
```

raw builder에서도 같은 blob shard를 만들 수 있습니다.

```python
manifest_path = builder.build_blob_shards(require_all=True)
```

### Dense-long shard

새 포맷입니다. 모든 `(feature_id, sample_id)` 조합을 parquet row로 저장하고, missing 여부는 `mask`로 표현합니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8   # 1=present, 0=missing
value       Float64 # mask=0이면 무시
```

물리 정렬은 `feature_id asc, sample_id asc`입니다. dense-long writer의 기본 row group은 feature 128개 단위입니다. feature 하나만 조회해도 해당 feature가 들어 있는 128개 feature 묶음을 읽지만, row group 수와 metadata overhead가 줄어 파일 크기와 build 시간이 개선됩니다. build 중에는 feature-major memmap을 채우고 같은 메모리 상태에서 selection stats를 계산하므로, stats 생성을 위해 최종 parquet를 다시 스캔하지 않습니다.

```python
dense_manifest = builder.build_dense_long_shards(
    out_dir="data/scalar_dense_long",
    require_all=True,
)

from scalar_feature_shard import open_dense_long_shard

with open_dense_long_shard(dense_manifest) as ds:
    values, valid = ds.load_feature_by_id(123)
    sample_values, sample_valid = ds.load_sample_by_id(10)
    top = ds.top_features_from_stats("y", top_k=256)
```

## 포맷 선택 기준

- blob shard는 파일 크기와 feature batch 조회에 유리합니다.
- dense-long shard는 표준 parquet로 디버깅하기 쉽고 sample 기준 조회가 유리합니다.
- sparse long 포맷은 현재 권장하지 않습니다. missing row를 생략하면 조회 시 dense vector를 다시 채워야 하고, 이번 구현 목표에서는 제외했습니다.

## 테스트와 벤치

```bash
python python/scripts/run_scalar_package_tests.py
```

패키지 회귀 테스트는 raw random-order stage, 기존 blob shard, dense-long shard, reader, precomputed selection stats 경로를 함께 검증합니다. 대형 성능 비교용 임시 benchmark 스크립트와 `data/tmp_*` 산출물은 저장소에 남기지 않습니다.

## 예제와 서버

Python end-to-end 예제:

- `examples/build_scalar_dense_long_example.py`

조회 서버:

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 scalar endpoint:

- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

`scalar_format`은 기본값 `auto`입니다. dense-long manifest는 `format=scalar-dense-long-shard-v1`로 자동 인식하고, 기존 blob shard manifest는 blob reader로 엽니다.
