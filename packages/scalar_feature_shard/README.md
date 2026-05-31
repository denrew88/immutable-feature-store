# scalar-feature-shard

scalar dense feature matrix를 Parquet 기반 dense-long shard로 만들고 조회하는 Python 패키지입니다.

현재 scalar 쪽 최종 shard 포맷은 dense-long 하나만 지원합니다. 예전 feature-major scalar API와 reader는 제거되었습니다.

## Public API

- `write_sample_meta(...)`, `write_feature_meta(...)`: dense `sample_id`, `feature_id` 기준 metadata parquet를 작성합니다.
- `ScalarDatasetBuilder`: sample을 순차적으로 받아 bundle stage를 만들고 최종 dense-long shard를 생성합니다.
- `ScalarRawDatasetBuilder`: sample별 raw parquet를 임의 순서로 만든 뒤, 마지막에 dense-long shard로 materialize합니다.
- `build_shard(...)`: sample-bundle manifest 또는 path column이 있는 sample metadata에서 dense-long shard를 생성합니다.
- `build_dense_long_shards_from_sample_bundles(...)`: bundle/raw sample rows에서 dense-long shard를 직접 생성합니다.
- `open_dense_long_shard(...)`: dense-long shard reader를 엽니다.
- `select_features(...)`, `run_selection(...)`: `selection_stats/<y>.parquet` 기반 feature selection을 수행합니다.

## Dense ID 규칙

- `sample_id`는 `sample_meta.parquet`의 row index입니다.
- `feature_id`는 `feature_meta.parquet`의 row index입니다.
- `sample_key`, `feature_key`는 외부 식별자이고, 저장과 조회의 물리 기준은 dense id입니다.
- scalar missing value는 최종 dense-long shard에서 `mask=0`으로 표현합니다.

## Sequential Builder

순차 builder는 sample을 `status().next_expected_sample_id`부터 순서대로 받아야 합니다. 중간에 멈추면 stage manifest와 bundle parquet를 기준으로 이어서 쓸 수 있습니다.

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder

builder = ScalarDatasetBuilder.open_session(
    out_dir="data/scalar_dense_long",
    sample_meta_path="data/sample_meta.parquet",
    feature_meta_path="data/feature_meta.parquet",
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
)

status = builder.status()
for sample_id in range(status.next_expected_sample_id, n_samples):
    builder.write_sample(sample_id, {"feature_a": 1.23, "feature_b": None})

manifest_path = builder.build_shards()
```

## Random Raw-Sample Builder

raw builder는 sample별 parquet를 따로 commit하므로 sample 순서 제약이 없습니다. supervisor는 `pending_sample_ids()`로 남은 sample을 확인한 뒤 worker에게 나눠줄 수 있습니다.

```python
from scalar_feature_shard import BuildOptions, ScalarRawDatasetBuilder

builder = ScalarRawDatasetBuilder.open_session(
    out_dir="data/scalar_raw_stage",
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
manifest_path = builder.build_dense_long_shards(
    out_dir="data/scalar_dense_long",
    require_all=True,
)
```

raw stage 파일 구조:

```text
scalar_raw_stage/
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  raw_samples/
    sample_000000000003.parquet
    sample_000000000007.parquet
```

raw sample parquet schema:

```text
sample_id   Int64
feature_id  Int32
value       Float64
```

## Dense-Long Shard Format

최종 shard는 모든 `(feature_id, sample_id)` 조합을 row로 저장합니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8   # 1=present, 0=missing
value       Float64 # mask=0이면 무시
```

물리 정렬은 `feature_id asc, sample_id asc`입니다. 기본 row group은 feature 128개 단위라서 feature 단위 조회에서 row-group pruning을 기대할 수 있고, 지나치게 작은 row group으로 인한 metadata overhead도 줄입니다.

artifact 구조:

```text
scalar_dense_long_shard/
  dense_long_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  dense_long_parts/
    part_0000.parquet
    part_0001.parquet
  selection_stats/
    y.parquet
```

## Reader

```python
from scalar_feature_shard import open_dense_long_shard

with open_dense_long_shard("data/scalar_dense_long/dense_long_shard_manifest.json") as ds:
    values, valid = ds.load_feature_by_id(123)
    sample_values, sample_valid = ds.load_sample_by_id(10)
    keyed = ds.get_values_by_key("feature_a", ["sample_000001", "sample_000002"])
    top = ds.top_features_from_stats("y", top_k=256)
```

## Selection

build option의 `stats_y_cols`에 target column을 지정하면 build 시점에 `selection_stats/<y>.parquet`가 만들어집니다. selection은 이 precomputed stats를 사용합니다.

```python
from scalar_feature_shard import SelectionOptions, select_features

result = select_features(
    "data/scalar_dense_long/dense_long_shard_manifest.json",
    y_col="y",
    options=SelectionOptions(top_m=256),
)
```

## API Server

권장 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다. scalar는 dense-long manifest만 엽니다.

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 scalar endpoint:

- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

## Tests

```powershell
python python\scripts\run_scalar_package_tests.py
```
