# Scalar Dense-Long Shard Format

scalar shard는 dense `feature x sample` 행렬을 Parquet로 저장하고 조회하기 위한 포맷입니다. 현재 최종 shard 포맷은 dense-long 하나만 지원합니다.

## 전체 흐름

1. `sample_meta.parquet`, `feature_meta.parquet`를 먼저 준비합니다.
2. builder가 sample별 값을 받습니다.
3. 값은 순차 bundle stage 또는 random raw-sample stage에 저장됩니다.
4. shard build 단계에서 stage 파일들을 읽어 dense-long part parquet를 만듭니다.
5. reader와 API 서버는 `dense_long_shard_manifest.json`을 기준으로 metadata, locator, part parquet를 연결해 조회합니다.

scalar 값은 숫자 하나입니다. missing 값은 row를 생략하지 않고 `mask=0`으로 저장합니다.

## Metadata

`sample_id`와 `feature_id`는 dense id입니다.

- `sample_id`: `sample_meta.parquet`의 row index
- `feature_id`: `feature_meta.parquet`의 row index

`sample_key`, `feature_key`는 외부 식별자입니다. key 조회가 들어오면 reader가 metadata에서 dense id로 변환한 뒤 실제 part를 조회합니다.

## Random Raw-Sample Stage

random builder는 sample 하나를 parquet 하나로 commit합니다. sample 작성 순서 제약이 없기 때문에 외부 worker가 sample을 나눠 처리하기 쉽고, 중간에 멈춰도 이미 commit된 sample은 다시 만들 필요가 없습니다.

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

raw sample parquet schema:

```text
sample_id   Int64
feature_id  Int32
value       Float64
```

commit 규칙:

- 먼저 `.tmp` 파일에 씁니다.
- 정상 종료 후 최종 `.parquet` 파일로 rename합니다.
- commit log인 `raw_samples.jsonl`에 sample id와 파일 경로를 기록합니다.
- `pending_sample_ids()`는 metadata 전체 sample 중 아직 commit log에 없는 sample을 반환합니다.

`finish_stage()`는 raw parquet 파일을 다시 쓰지 않습니다. raw sample 파일 목록을 `sample_major_manifest.json`으로 연결해서 shard build 입력으로 만듭니다.

## Dense-Long Shard

최종 artifact 구조:

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
- 모든 `(feature_id, sample_id)` 조합이 row로 존재합니다.

물리 정렬:

```text
feature_id asc, sample_id asc
```

이 정렬은 feature 단위 조회를 빠르게 만들기 위한 기준입니다. feature 하나를 읽을 때 `feature_id` filter로 row group pruning을 기대할 수 있습니다.

## Part와 Row Group

part 파일은 `target_shard_mb`/`targetShardBytes` 기준으로 나뉩니다. 하나의 part에는 연속된 feature id 구간이 들어갑니다.

row group은 기본적으로 feature 128개 단위입니다.

```text
row_group_rows = n_samples * dense_long_row_group_features
```

feature 1개 조회 시 최대 128개 feature 묶음이 같은 row group에 들어갈 수 있습니다. 대신 row group 수가 줄어 metadata overhead와 파일 크기, build 시간이 줄어듭니다. 현재 기본값은 실험적으로 file size/build/query 균형이 좋았던 128입니다.

## Build 구현 방식

dense-long build는 다음 순서로 동작합니다.

1. sample metadata와 feature metadata를 읽어 전체 행렬 크기를 결정합니다.
2. feature-major backing array를 준비합니다.
3. stage parquet들을 순회하며 `(sample_id, feature_id, value)`를 feature-major 위치에 채웁니다.
4. 값이 없는 위치는 valid mask가 false인 상태로 둡니다.
5. 같은 backing array에서 selection stats를 계산합니다.
6. feature id 구간별로 dense-long rows를 만들어 part parquet로 씁니다.
7. part 위치를 `feature_locator.parquet`에 기록합니다.
8. 전체 연결 정보를 `dense_long_shard_manifest.json`에 기록합니다.

feature-major backing array를 쓰는 이유는 selection stats와 최종 dense-long write가 모두 feature 단위 접근을 많이 하기 때문입니다. 최종 parquet는 long format이지만, build 중간 상태는 feature-major가 더 효율적입니다.

## Feature Locator

`feature_locator.parquet`는 feature id가 어느 part와 offset에 있는지 알려줍니다.

```text
feature_id       Int32
part_id          Int32
offset_in_part   Int64
```

`offset_in_part`는 해당 part 안에서 feature가 시작하는 feature offset입니다. dense-long part는 feature-major 정렬이므로 reader는 locator를 보고 필요한 part만 열 수 있습니다.

## Selection Stats

`BuildOptions.stats_y_cols` 또는 `BuildShardConfig.statsYCols`에 target column을 지정하면 build 중에 selection stats를 만듭니다.

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
n_y_overlap      Int64
```

selection은 최종 parquet 전체를 다시 훑지 않고 이 stats 파일을 사용합니다.

## 조회 서버

권장 조회 서버:

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

scalar endpoint:

- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

id와 key는 같은 축에서 둘 중 하나만 허용합니다. 예를 들어 `sample_id`와 `sample_key`를 동시에 주면 에러입니다.

## Validation

reader/build 검증에서 확인해야 할 내용:

- manifest의 `format`은 `scalar-dense-long-shard-v1`입니다.
- part parquet schema는 `feature_id`, `sample_id`, `mask`, `value`입니다.
- part rows는 `feature_id asc, sample_id asc`로 정렬되어야 합니다.
- 각 feature에는 정확히 `n_samples` rows가 있어야 합니다.
- `mask=0`인 row의 `value`는 조회 결과에서 missing으로 처리해야 합니다.
