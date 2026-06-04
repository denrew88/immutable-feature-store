# Scalar Dense-Long Shard Format

scalar shard는 scalar feature 값을 Parquet 기반 dense-long layout으로 저장하는 포맷입니다. 현재 scalar 쪽 최종 shard 포맷은 dense-long 하나만 지원합니다.

## 전체 흐름

1. `sample_meta.parquet`와 `feature_meta.parquet`를 준비합니다.
2. `ScalarDatasetBuilder`를 열고 `status().pending_sample_ids` 또는 Java의 `status().pendingSampleIds`를 확인합니다.
3. 각 sample에 대해 `write_sample(...)` 또는 `writeSample(...)`로 feature 값을 씁니다. sample id 순서는 강제하지 않습니다.
4. sample 하나가 완료되면 builder가 `raw_samples/sample_*.parquet.tmp`를 만들고, 정상 종료 후 `.parquet`로 rename한 뒤 `raw_samples.jsonl`에 commit record를 남깁니다.
5. `finish_stage()`는 완료된 raw sample 파일 목록을 `sample_major_manifest.json`으로 확정합니다. 이 단계는 raw parquet를 다시 쓰지 않습니다.
6. `build_shards(...)` 또는 `buildDenseLongShards(...)`가 raw sample long rows를 읽어 최종 dense-long parquet part를 만듭니다.
7. reader나 API 서버는 `scalar_shard_manifest.json`을 entrypoint로 사용합니다.

순차 builder와 random/raw builder는 더 이상 분리하지 않습니다. 순차 실행은 `pending_sample_ids`를 앞에서부터 처리하는 특수한 사용 방식일 뿐입니다.

## Metadata

`sample_id`와 `feature_id`는 dense row id입니다.

- `sample_id`: `sample_meta.parquet`의 row index입니다.
- `feature_id`: `feature_meta.parquet`의 row index입니다.
- `sample_key`, `feature_key`: 외부 key입니다. 조회 API에서 key를 받으면 metadata에서 dense id로 변환한 뒤 part를 조회합니다.

selection stats를 만들려면 `sample_meta.parquet`에 target column이 있어야 합니다. Python은 `BuildOptions.stats_y_cols`, Java는 `BuildShardConfig.statsYCols`로 target column을 지정합니다.

## Raw Sample Stage

builder stage의 파일 구조는 다음과 같습니다.

```text
scalar_stage/
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  raw_samples/
    sample_000000000000.parquet
    sample_000000000017.parquet
```

`raw_samples/*.parquet` schema:

```text
sample_id   Int64
feature_id  Int32
value       Float64
```

missing 값은 raw row를 쓰지 않는 방식으로 표현합니다. `None` 또는 `NaN`은 missing으로 처리됩니다.

commit 규칙:

- 먼저 `.tmp` 파일을 씁니다.
- sample 파일이 정상적으로 생성되면 최종 `.parquet` 파일로 atomic rename합니다.
- `raw_samples.jsonl`에 `sample_id`, `sample_key`, relative path, row count를 기록합니다.
- resume 시에는 commit log에 있고 실제 parquet 파일도 존재하는 sample만 완료로 인정합니다.
- `pending_sample_ids()`는 metadata 전체 sample 중 아직 commit되지 않은 sample id 목록을 반환합니다.

이 구조 덕분에 외부 DB 조회나 전처리가 비싼 경우에도 이미 완료된 sample은 다시 만들 필요가 없습니다. 여러 worker가 서로 다른 sample id를 맡아 raw sample을 만들고, 마지막 compact/build 단계만 하나의 supervisor가 실행하는 식으로 사용할 수 있습니다.

`sample_major_manifest.json`은 shard build 단계의 입력 목록입니다.

```json
{
  "format": "scalar-sample-major-v1",
  "version": 1,
  "sample_meta_path": "sample_meta.parquet",
  "feature_meta_path": "feature_meta.parquet",
  "sample_paths": [
    "raw_samples/sample_000000000000.parquet",
    "raw_samples/sample_000000000017.parquet"
  ],
  "sample_ids": [0, 17],
  "sample_id_col": "sample_id",
  "feature_id_col": "feature_id",
  "value_col": "value"
}
```

`sample_paths`는 완료된 raw sample parquet 목록입니다. `sample_ids`는 파일과 sample id의 대응을 명시하는 보조 정보입니다. 표준 builder가 만든 raw file에는 항상 `sample_id` column이 있고, shard build는 이 column을 기준으로 dense-long row를 채웁니다.

## Dense-Long Shard

최종 artifact 구조:

```text
scalar_shard/
  scalar_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  parts/
    part_0000.parquet
    part_0001.parquet
  selection_stats/
    y.parquet
```

`parts/*.parquet` schema:

```text
feature_id  Int32
sample_id   Int64
mask        UInt8
value       Float64
```

의미:

- `mask=1`: 해당 `(feature_id, sample_id)` 값이 present입니다.
- `mask=0`: missing입니다. 이때 `value`는 조회 결과에서 무시합니다.
- 모든 `(feature_id, sample_id)` 조합이 row로 존재합니다. 즉 sparse long이 아니라 dense long입니다.

물리 정렬은 `feature_id asc, sample_id asc`입니다. feature 단위 조회가 많기 때문에 feature-major 정렬을 사용합니다.

## Part와 Row Group

part 파일은 `target_shard_mb` 또는 Java의 `targetShardBytes`를 기준으로 나뉩니다. 하나의 part에는 연속된 feature id 구간이 들어갑니다.

row group은 기본적으로 feature 128개 단위입니다.

```text
row_group_rows = n_samples * dense_long_row_group_features
```

row group을 feature 1개 단위로 만들면 pruning은 더 세밀하지만 metadata overhead와 파일 수/크기 부담이 커집니다. 현재 기본값 128은 테스트 기준으로 파일 크기, build 시간, 조회 시간의 균형이 좋았던 값입니다.

## Build 구현 방식

dense-long build는 다음 순서로 동작합니다.

1. sample metadata와 feature metadata를 읽어 전체 matrix 크기를 결정합니다.
2. raw sample parquet의 `(sample_id, feature_id, value)` rows를 읽습니다.
3. 내부 feature-major 배열에 present value와 valid mask를 채웁니다.
4. 값이 없는 위치는 valid mask false로 둡니다.
5. 지정된 target column이 있으면 selection stats를 계산합니다.
6. feature id 구간별로 dense-long rows를 만들어 part parquet로 씁니다.
7. feature id가 어느 part의 어느 offset에 있는지 `feature_locator.parquet`에 기록합니다.
8. 최종 연결 정보를 `scalar_shard_manifest.json`에 기록합니다.

최종 parquet는 long format이지만 build 중간 배열은 feature-major입니다. selection stats 계산과 최종 part write가 모두 feature 단위 접근을 많이 하기 때문입니다.

## Feature Locator

`feature_locator.parquet`는 reader가 필요한 part만 읽도록 돕는 lookup table입니다.

```text
feature_id       Int32
global_rank      Int32
part_id          Int32
offset_in_part   Int64
first_row_in_part Int64
```

`global_rank`는 현재 dense feature id와 같습니다. `offset_in_part`는 해당 part 안에서 feature가 시작되는 feature offset이고, `first_row_in_part`는 그 feature의 첫 row 위치입니다. dense-long part가 feature-major 정렬이므로 reader는 locator를 보고 필요한 part를 빠르게 고를 수 있습니다.

## Selection Stats

`stats_y_cols`를 지정하면 build 중 다음 파일을 생성합니다.

```text
selection_stats/
  y.parquet
```

schema:

```text
feature_id       Int32
part_id          Int32
offset_in_part   Int64
r2y              Float64
n_y_overlap      Int32
```

feature selection은 최종 dense-long parquet 전체를 다시 스캔하지 않고 이 stats 파일을 사용합니다.

## Python 사용 예

```python
from scalar_feature_shard import BuildOptions, ScalarDatasetBuilder, open_dense_long_shard

with ScalarDatasetBuilder.open_session(
    out_dir="data/scalar_stage",
    sample_meta_path="data/sample_meta.parquet",
    feature_meta_path="data/feature_meta.parquet",
    build_options=BuildOptions(target_shard_mb=32, stats_y_cols=("y",)),
) as builder:
    for sample_id in builder.pending_sample_ids():
        values = {
            "feature_a": 1.23,
            "feature_b": None,  # missing
        }
        builder.write_sample(sample_id, values, skip_if_completed=True)

    manifest_path = builder.build_shards(require_all=True)

with open_dense_long_shard(manifest_path) as ds:
    values, valid = ds.load_feature_by_key("feature_a")
```

## Java 사용 예

```java
BuildShardConfig cfg = new BuildShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.statsYCols = java.util.Arrays.asList("y");

String manifestPath;
try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
        "data/scalar_stage",
        "data/sample_meta.parquet",
        "data/feature_meta.parquet",
        null,
        cfg)) {
    for (Long sampleId : builder.status().pendingSampleIds) {
        Map<Object, Object> values = new LinkedHashMap<Object, Object>();
        values.put("feature_a", 1.23);
        values.put("feature_b", null);
        builder.writeSample(sampleId.longValue(), values, true);
    }
    manifestPath = builder.buildShards(true);
}
```

## Config Guide

처음 사용하는 경우 scalar dense-long build에는 아래 정도만 넣으면 됩니다.

```python
BuildOptions(
    target_shard_mb=32,
    stats_y_cols=("y",),
)
```

Java에서는 같은 의미로 아래처럼 씁니다.

```java
BuildShardConfig cfg = new BuildShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.statsYCols = java.util.Arrays.asList("y");
```

### BuildOptions / BuildShardConfig

| Python | Java | 기본값 | 언제 바꾸는가 |
| --- | --- | --- | --- |
| `target_shard_mb` | `targetShardBytes` | 32MB | dense-long part 파일이 너무 많으면 키우고, 한 파일이 너무 커서 query/build 부담이 크면 줄입니다. |
| `stats_y_cols` | `statsYCols` | `None` / `null` | feature selection을 할 target column 목록입니다. 보통 `("y",)` 또는 `["y"]`를 넣습니다. |
| `y_col` | `yCol` | `"y"` | `stats_y_cols`를 생략했을 때 사용할 단일 target column입니다. |
| `sample_key_col` | `sampleKeyCol` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `feature_key_col` | `featureKeyCol` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |
| `feature_id_col` | `featureIdCol` | `"feature_id"` | raw/sample-major 입력 schema가 기본값과 다를 때만 바꿉니다. |
| `sample_id_col` | `sampleIdCol` | `"sample_id"` | raw/sample-major 입력 schema가 기본값과 다를 때만 바꿉니다. |
| `value_col` | `valueCol` | `"value"` | raw/sample-major 입력의 value column 이름이 다를 때만 바꿉니다. |
| `path_col` | `pathCol` | `"sample_path"` | sample-major manifest/table의 path column 이름이 다를 때만 바꿉니다. |
| `values_dtype` | `valuesType` | `float64` / `FLOAT64` | 현재 dense-long 표준은 float64입니다. 특별한 이유가 없으면 바꾸지 않습니다. |
| `valid_dtype` | `validType` | `uint8` / `UINT8` | mask column 타입입니다. 특별한 이유가 없으면 바꾸지 않습니다. |
| 없음 | `denseLongRowGroupFeatures` | 128 | parquet row group 하나에 묶을 feature 수입니다. query pruning과 파일 크기의 절충값입니다. |
| 없음 | `denseLongPartFeatures` | 0 | part 하나의 feature 수를 강제로 고정할 때만 씁니다. 0이면 target size 기준 자동 계산입니다. |

권장값:

- 일반 build: `target_shard_mb=32`, `stats_y_cols=("y",)`
- selection stats가 필요 없으면: `stats_y_cols=()`, 단 현재 Java는 비어 있으면 `yCol` 하나를 만들기 때문에 y column이 없는 metadata에서는 stats 관련 설정을 조심해야 합니다.
- sample/feature key 이름이 기본값이면: `sample_key_col`, `feature_key_col`은 건드리지 않습니다.
- row group tuning은 성능 테스트 후에만 합니다. 현재 기본 128은 파일 크기, build 시간, 조회 시간을 함께 본 절충값입니다.

### SelectionOptions / SelectionConfig

feature selection은 보통 아래처럼 `y_col`과 `top_m`만 지정합니다.

```python
SelectionOptions(y_col="y", top_m=256)
```

| Python | Java | 기본값 | 의미 |
| --- | --- | --- | --- |
| `y_col` | 별도 인자/통계 이름 | `"y"` | 사용할 `selection_stats/<y>.parquet` target입니다. |
| `top_m` | `topM` | 100 | 최종 선택할 feature 수입니다. |
| `y_r2_threshold` | `yR2Threshold` | 0.01 | y와의 최소 R^2입니다. 낮을수록 후보가 많아집니다. |
| `min_non_null_y` | `minNonNullY` | 200 | feature와 y가 동시에 present인 sample 최소 개수입니다. |
| `ff_r2_threshold` | `ffR2Threshold` | 0.9 | 이미 선택된 feature와 너무 비슷한 후보를 제거하는 기준입니다. |
| `min_non_null_pair` | `minNonNullPair` | 200 | feature-feature R^2 계산에 필요한 공통 present sample 수입니다. |
| `initial_cap` | `initialCap` | 2048 | 처음 가져올 후보 수입니다. |
| `max_step` | `maxStep` | 4096 | 후보가 부족할 때 한 번에 늘리는 최대 후보 수입니다. |
| `batch_size` | `batchSize` | Python 512 / Java 1024 | 후보 feature를 reader에서 읽는 batch 크기입니다. |
| `max_gap` | `maxGap` | Python 64 / Java 0 | 추가 후보 탐색 중 개선 없이 허용할 gap입니다. 0은 제한 없음으로 취급합니다. |
| `max_candidates` | `maxCandidates` | 0 | 후보 수 상한입니다. 0이면 제한하지 않습니다. |

## Java Builder Value API

Java가 외부 시스템에서 값을 받아 builder에 쓰는 경우, 예제 서버는 `python/scripts/serve_synthetic_value_api.py`를 사용합니다. 이 서버는 최종 dense-long shard 조회 API가 아니라, Java builder가 sample별 scalar 값을 받아오기 위한 value source 예제입니다.

실행:

```powershell
python python\scripts\serve_synthetic_value_api.py --host 127.0.0.1 --port 8010
```

### `POST /scalar/values`

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
| `missing_rate` | float | no | missing scalar value 비율 |
| `n_latent_groups` | int | no | synthetic latent group 수 |
| `noise_scale` | float | no | synthetic noise scale |

요청 예:

```json
{
  "sample_meta_path": "data/sample_meta.parquet",
  "feature_meta_path": "data/feature_meta.parquet",
  "sample_id": 0,
  "feature_ids": [0, 1, 2],
  "seed": 7,
  "missing_rate": 0.1,
  "n_latent_groups": 16,
  "noise_scale": 0.25
}
```

응답은 sample 하나에 대한 scalar value 목록입니다. `present=false`이면 `value`는 `null`입니다.

```json
{
  "sample_id": 0,
  "sample_key": "sample_000000",
  "feature_count": 3,
  "values": [
    {
      "feature_id": 0,
      "feature_key": "feature_000000",
      "present": true,
      "value": 1.23
    },
    {
      "feature_id": 1,
      "feature_key": "feature_000001",
      "present": false,
      "value": null
    }
  ]
}
```

`BuildScalarDenseLongFromValueApiMain`과 jar 예제 `BuildScalarDenseLongFromValueApiWithJarExample`는 `present=true`이고 `value`가 null이 아닌 항목만 `writeSample(...)`에 넘깁니다. 나머지는 raw row를 만들지 않고, dense-long materialize 단계에서 `mask=0` missing row로 채워집니다.

## 조회 API 서버

권장 조회 서버는 `python/scripts/serve_feature_query_api.py`입니다.

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

주요 endpoint:

- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

id와 key는 같은 축에서 둘 중 하나만 허용합니다. 예를 들어 `sample_id`와 `sample_key`를 동시에 주면 에러입니다.

## Java Runtime Dependency

`scalar-feature-shard-java`는 thin jar입니다. 실행 시 package jar와 함께 다음 runtime jar가 classpath에 있어야 합니다.

- DuckDB JDBC: raw sample parquet, dense-long part parquet, selection stats parquet 생성과 조회를 담당합니다. zstd 압축도 DuckDB parquet writer가 처리합니다.
- Jackson: `raw_state.json`, `sample_major_manifest.json`, `scalar_shard_manifest.json` 같은 JSON 파일을 읽고 씁니다.

Hadoop/Parquet Java writer, Arrow, SLF4J, Woodstox, stax2, commons jar는 현재 scalar dense-long 구현에 필요하지 않습니다.

## Validation

구현과 데이터 검증 시 확인할 항목:

- manifest의 `format`은 `scalar-dense-long-shard-v1`입니다.
- part parquet schema는 `feature_id`, `sample_id`, `mask`, `value`입니다.
- part rows는 `feature_id asc, sample_id asc`로 정렬되어야 합니다.
- 각 feature에는 정확히 `n_samples` rows가 있어야 합니다.
- `mask=0` row는 조회 결과에서 missing으로 처리해야 합니다.
