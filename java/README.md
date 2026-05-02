Java 8 기준 빌드/실행 안내입니다. Maven은 사용하지 않습니다.

## 준비물

- Java 8
- Java 8용 DuckDB JDBC jar: `duckdb_jdbc-1.1.3.jar`

jar는 저장소에 포함하지 않는다. 먼저 받아야 한다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

`java/lib` 아래에는 DuckDB jar를 하나만 두는 것이 좋습니다.

## 컴파일

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
javac -cp "java\lib\duckdb_jdbc-1.1.3.jar" -d java\out @java\sources.txt
```

## Scalar CLI

`sample_meta.parquet`와 `feature_meta.parquet`에서 바로 scalar shard 만들기:

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-meta C:\data\sample_meta.parquet ^
  --feature-meta C:\data\feature_meta.parquet ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

bundle 기반 sample-major stage에서 scalar shard 만들기:

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-bundle-manifest C:\data\sample_major_stage\sample_major_manifest.json ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

selection 실행:

```powershell
java -cp "java\lib\*;java\out" scripts.RunSelectionMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --top-m 100
```

scalar feature 하나 위치 확인:

```powershell
java -cp "java\lib\*;java\out" scripts.LocateFeatureMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --feature-id 12345
```

## Java Scalar Public API

진입점은 `fs.io.ScalarFeatureShards`입니다.

포함 기능:

- `loadManifest(...)`
- `open(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`
- `buildCandidates(...)`
- `selectFeatures(...)`

direct-ingestion builder는 `fs.io.ScalarDatasetBuilder`입니다.

주요 메서드:

- `writeSample(...)`
- `openSample(...)`
- `finishSampleMajor()`
- `updateFeatureMeta(...)`
- `buildShards(keepSampleMajor)`

## Array CLI

기존 bundle manifest에서 array shard 만들기:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.BuildArrayShardsMain ^
  --bundle-manifest C:\data\array_bundle_manifest.json ^
  --out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16
```

array feature 하나 위치 확인:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.LocateArrayFeatureMain ^
  --manifest C:\data\array_shards\array_binary_shard_manifest.json ^
  --feature-id 12345
```

synthetic array 데이터 생성 후 shard 빌드:

```powershell
java -cp "java\lib\duckdb_jdbc-1.1.3.jar;java\out" scripts.GenerateArraySynthMain ^
  --bundle-out-dir C:\data\array_bundles ^
  --sample-meta C:\data\array_sample_meta.parquet ^
  --shard-out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16 ^
  --n-samples 256 ^
  --n-features 64 ^
  --seed 7
```

## Java Array Public API

진입점은 `fs.io.ArrayBinaryShards`입니다.

포함 기능:

- `buildFromBundles(...)`
- `loadManifest(...)`
- `open(...)`
- `loadLocator(...)`
- `loadSampleIds(...)`
- `loadFeatureIds(...)`
- `writeSampleMeta(...)`
- `writeFeatureMeta(...)`
- `newBuilder(...)`

direct-ingestion builder는 `fs.io.ArrayDatasetBuilder`입니다.

## Array Binary v3 요약

자바 array shard는 Python array binary shard **v3** 포맷을 따릅니다.

핵심:

- standalone artifact 구조
  - `array_binary_shard_manifest.json`
  - `sample_meta.parquet`
  - `feature_meta.parquet`
  - 필요 시 `categorical_dictionaries/*.parquet`
  - `array_binary_feature_shards/shard_XXXX.blocks.idx`
  - `array_binary_feature_shards/shard_XXXX.blocks.bin`
- dense id
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- `point_schema`는 manifest가 정의
- `time`, `value`는 필수 아님
- categorical은 integer code로 저장하고 reader에서 label로 decode 가능

## 테스트

전체 Java 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunTestsMain --seed 0
```

array storage 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayStorageTestsMain
```

array synthetic 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArraySyntheticTestsMain
```

array v3 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayV3TestsMain
```

scalar builder 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
```

## 참고

- `java/lib/duckdb_jdbc-1.1.3.jar`는 git 추적 대상이 아닙니다.
- 현재 Java array binary codec 기본값은 `none`입니다.
- `blocks.idx`는 `(feature_id, block_id)` 슬롯마다 32바이트 고정 레코드를 가집니다.
- `blocks.bin`은 가변 길이 payload를 저장한다.
- array block 기본 크기는 `samples_per_block=16`입니다.
- shard 분할 기본 기준은 `target_shard_mb=32`입니다.
