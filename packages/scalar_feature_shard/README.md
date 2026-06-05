# scalar-feature-shard

`scalar-feature-shard`는 scalar feature matrix를 Parquet 기반 dense-long shard로 만들고 조회하는 Python 패키지입니다. 현재 최종 scalar shard 포맷은 dense-long 하나만 지원합니다.

## Public API

- `write_sample_meta(...)`, `write_feature_meta(...)`: dense `sample_id`, `feature_id` 기준 metadata parquet를 작성합니다.
- `ScalarDatasetBuilder`: sample별 raw parquet stage를 만들고 최종 dense-long shard로 materialize하는 표준 builder입니다. sample id 순서를 강제하지 않습니다.
- `build_shard(...)`: sample-major manifest나 sample metadata에서 dense-long shard를 생성합니다.
- `build_dense_long_shards_from_sample_major_manifest(...)`: raw sample rows 목록이 담긴 sample-major manifest에서 dense-long shard를 직접 생성합니다.
- `open_dense_long_shard(...)`: dense-long shard reader를 엽니다.
- `select_features(...)`, `run_selection(...)`: `selection_stats/<y>.parquet` 기반 feature selection을 수행합니다.

## Builder 사용

순차 builder와 raw builder는 분리하지 않습니다. `ScalarDatasetBuilder` 하나가 resume 가능한 sample-file stage를 담당합니다. 순차 실행을 원하면 `pending_sample_ids()`를 앞에서부터 처리하면 됩니다.

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder

with ScalarDatasetBuilder.open_session(
    out_dir="data/scalar_stage",
    sample_meta_path="data/sample_meta.parquet",
    feature_meta_path="data/feature_meta.parquet",
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
) as builder:
    for sample_id in builder.pending_sample_ids():
        builder.write_sample(
            sample_id,
            {"feature_a": 1.23, "feature_b": None},
            skip_if_completed=True,
        )
    manifest_path = builder.build_shards(require_all=True)
```

stage 파일 구조:

```text
scalar_stage/
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  raw_samples/
    sample_000000000000.parquet
```

`raw_samples.jsonl`에 기록되고 실제 parquet 파일도 존재하는 sample만 완료로 인정합니다. 중간에 중단되면 같은 `out_dir`로 session을 다시 열고 `pending_sample_ids()`만 처리하면 됩니다.

## Config Guide

처음에는 아래 설정만 넣으면 됩니다.

```python
BuildOptions(target_shard_mb=32, stats_y_cols=("y",))
```

| option | 기본값 | 설명 |
| --- | --- | --- |
| `target_shard_mb` | 32 | dense-long part 하나의 목표 크기입니다. part가 너무 많으면 키우고, 한 파일이 너무 크면 줄입니다. |
| `stats_y_cols` | `None` | selection stats를 만들 target column 목록입니다. feature selection을 하려면 보통 `("y",)`를 넣습니다. |
| `y_col` | `"y"` | `stats_y_cols`를 생략했을 때 사용할 단일 target column입니다. |
| `sample_key_col` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `feature_key_col` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |
| `sample_id_col`, `feature_id_col`, `value_col` | 기본 schema | raw/sample-major parquet column 이름을 바꿨을 때만 수정합니다. |
| `values_dtype`, `valid_dtype` | `float64`, `uint8` | dense-long 표준 타입입니다. 보통 바꾸지 않습니다. |

selection은 보통 아래처럼 씁니다.

```python
SelectionOptions(y_col="y", top_m=256)
```

`top_m`은 최종 선택 개수입니다. `y_r2_threshold`는 y와의 최소 R^2, `ff_r2_threshold`는 이미 선택된 feature와의 중복 제거 기준입니다. 나머지 batch/cap 계열 값은 후보가 너무 많거나 selection 속도 병목이 보일 때만 조정합니다.

## Dense-Long Shard

최종 part parquet는 모든 `(feature_id, sample_id)` 조합을 row로 갖습니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8   # 1=present, 0=missing
value       Float64 # mask=0이면 내부 parquet에서는 NaN, reader/API에서는 missing/null
```

물리 정렬은 `feature_id asc, sample_id asc`입니다. 기본 row group은 feature 128개 단위입니다.

artifact 구조:

```text
scalar_shard/
  scalar_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  parts/
    part_0000.parquet
  selection_stats/
    y.parquet
```

## Reader

```python
from scalar_feature_shard import open_dense_long_shard

with open_dense_long_shard("data/scalar_stage/scalar_shard/scalar_shard_manifest.json") as ds:
    values, valid = ds.load_feature_by_key("feature_a")
    sample_values, sample_valid = ds.load_sample_by_id(10)
    top = ds.top_features_from_stats("y", top_k=256)
```

## Selection

build option의 `stats_y_cols`에 target column을 지정하면 build 시 `selection_stats/<y>.parquet`가 생성됩니다. selection은 이 precomputed stats를 사용합니다.

```python
from scalar_feature_shard import SelectionOptions, select_features

result = select_features(
    "data/scalar_stage/scalar_shard/scalar_shard_manifest.json",
    options=SelectionOptions(y_col="y", top_m=256),
)
```

## API Server

권장 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다.

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 scalar endpoint:

- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

자세한 포맷과 구현 방식은 [docs/scalar_parquet_shard_format.md](../../docs/scalar_parquet_shard_format.md)를 참고하십시오.

## Tests

```powershell
python python\scripts\run_scalar_package_tests.py
python python\scripts\run_builder_session_tests.py
```
