# immutable-feature-store

offline에서 만든 feature 데이터를 빠르게 조회하고, 필요할 때 일반 Parquet 도구로도 디버깅할 수 있게 만든 저장 포맷과 Python/Java 구현입니다.

주요 포맷:

- **array binary shard**: trace 데이터를 feature-major custom binary로 저장하는 빠른 serving 포맷입니다.
- **array sample parquet**: trace 데이터를 sample-major long Parquet로 저장하는 viewer/debugging 친화 포맷입니다.
- **scalar dense-long shard**: scalar `feature x sample` 값을 dense-long Parquet로 저장하고 selection stats를 함께 만드는 포맷입니다.

## Directory Layout

```text
docs/
  array_binary_shard_format_v3.md
  array_sample_parquet_format_v1.md
  scalar_parquet_shard_format.md

python/
  fs/
  scripts/

java/
  src/
  README.md

packages/
  array_binary_shard/
  array_sample_parquet/
  scalar_feature_shard/
  array_binary_shard_java/
  array_sample_parquet_java/
  scalar_feature_shard_java/
```

## Formats

### Array Binary Shard

- 문서: [docs/array_binary_shard_format_v3.md](docs/array_binary_shard_format_v3.md)
- Python wheel: [packages/array_binary_shard](packages/array_binary_shard)
- Java jar: [packages/array_binary_shard_java](packages/array_binary_shard_java)

feature-major custom binary 포맷입니다. `blocks.idx`와 `blocks.bin` 묶음으로 shard를 구성하며, 빠른 random access serving을 목표로 합니다.

### Array Sample Parquet

- 문서: [docs/array_sample_parquet_format_v1.md](docs/array_sample_parquet_format_v1.md)
- Python wheel: [packages/array_sample_parquet](packages/array_sample_parquet)
- Java jar: [packages/array_sample_parquet_java](packages/array_sample_parquet_java)

sample-major long Parquet 포맷입니다. builder는 sample별 raw parquet를 먼저 만들고, compact 단계에서 `sample_parts/`와 `trace_index_parts/`로 묶습니다. 최종 point row 정렬은 `(sample_id, feature_id, point_idx)`입니다.

### Scalar Dense-Long Shard

- 문서: [docs/scalar_parquet_shard_format.md](docs/scalar_parquet_shard_format.md)
- Python wheel: [packages/scalar_feature_shard](packages/scalar_feature_shard)
- Java jar: [packages/scalar_feature_shard_java](packages/scalar_feature_shard_java)

scalar의 최종 shard는 dense-long 하나만 지원합니다. 모든 `(feature_id, sample_id)` 조합을 row로 저장하고 missing은 `mask=0`으로 표현합니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8
value       Float64
```

## Config Quick Start

처음에는 아래 설정만 기억하면 됩니다. 자세한 필드 설명은 각 포맷 문서의 `Config Guide`를 보십시오.

| 포맷 | 최소 권장 설정 | 설명 |
| --- | --- | --- |
| scalar dense-long | `BuildOptions(target_shard_mb=32, stats_y_cols=("y",))` | `target_shard_mb`는 part 목표 크기, `stats_y_cols`는 selection stats target입니다. |
| array sample parquet | `ArraySampleParquetBuildOptions(target_part_bytes=128 * 1024 * 1024, compression="zstd")` | viewer/debugging용 표준 array parquet입니다. part 크기와 압축만 주면 됩니다. |
| array custom binary | `ArrayBinaryBuildOptions(samples_per_block=16, target_shard_mb=32, codec="none")` | 빠른 feature-major serving용 특수 포맷입니다. 유지보수성이 중요하면 parquet를 우선 사용하십시오. |

공통적으로 `sample_key_col`, `feature_key_col`은 metadata column 이름이 기본값(`sample_key`, `feature_key`)과 다를 때만 바꾸면 됩니다.

## Python Server

권장 조회 서버:

```powershell
python python\scripts\serve_feature_query_api.py --host 127.0.0.1 --port 8000
```

지원 endpoint:

- `GET /healthz`
- `GET /cache-stats`
- `POST /array-sample-parquet/schema`
- `POST /array-sample-parquet/traces`
- `POST /scalar/schema`
- `POST /scalar/features`
- `POST /scalar/sample`
- `POST /scalar/top-features`

조회 요청은 dense id 또는 external key 중 하나만 받습니다. 같은 축에서 id와 key를 동시에 주면 에러입니다.

## Python Packages

### `array_binary_shard`

```powershell
cd packages\array_binary_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 API:

- `open_shard(...)`
- `ArrayDatasetBuilder`
- `build_shard(...)`

### `array_sample_parquet`

```powershell
cd packages\array_sample_parquet
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 API:

- `ArraySampleParquetDatasetBuilder.open_session(...)`
- `builder.status()`
- `builder.pending_sample_ids()`
- `builder.sample(...)`
- `sample.add_trace(...)`
- `builder.finish()`
- `builder.compact()`
- `open_array_sample_parquet(...)`

### `scalar_feature_shard`

```powershell
cd packages\scalar_feature_shard
python -m pip wheel . -w wheelhouse --no-deps --no-build-isolation
```

주요 API:

- `ScalarDatasetBuilder`
- `build_shard(...)`
- `build_dense_long_shards_from_sample_major_manifest(...)`
- `open_dense_long_shard(...)`
- `select_features(...)`

## Java

자세한 설명은 [java/README.md](java/README.md)를 참고하십시오.

필요한 runtime jar는 `java\download_java_libs.ps1`로 받습니다. 현재 Java package 3개 기준으로 Hadoop/Parquet Java writer stack은 필요하지 않습니다. `array-binary-shard-java`와 `scalar-feature-shard-java`는 DuckDB JDBC와 Jackson만 필요하고, `array-sample-parquet-java`는 raw sample write fast path 때문에 Arrow bridge jar가 추가로 필요합니다.

컴파일:

```powershell
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" -encoding UTF-8 -cp "java\lib\*" -d java\out @java\sources.txt
```

주요 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunArrayV3TestsMain
java -cp "java\lib\*;java\out" scripts.RunArraySampleParquetTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarNotebookBuilderTestsMain
```

## Examples

- [packages/array_sample_parquet/examples/build_array_sample_parquet_example.py](packages/array_sample_parquet/examples/build_array_sample_parquet_example.py)
- [packages/array_sample_parquet_java/examples/BuildArraySampleParquetWithJarExample.java](packages/array_sample_parquet_java/examples/BuildArraySampleParquetWithJarExample.java)
- [packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py](packages/scalar_feature_shard/examples/build_scalar_dense_long_example.py)
- [packages/scalar_feature_shard_java/examples/BuildScalarFeatureShardWithJarExample.java](packages/scalar_feature_shard_java/examples/BuildScalarFeatureShardWithJarExample.java)

## Data Policy

`data/` 아래 생성 산출물은 git에 올리지 않습니다. 저장소에는 코드, 문서, 패키지 scaffold, 작은 테스트/예제만 둡니다.
