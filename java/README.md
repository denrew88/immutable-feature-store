# Java 구현 안내

이 디렉터리에는 Java 8 기준 array/scalar builder, reader, test script, jar package build script가 들어 있습니다.

공통 원칙:

- array와 scalar 모두 dense `sample_id`, `feature_id`를 물리 id로 사용합니다.
- build API는 중단 후 재개 가능한 session 모델을 사용합니다.
- scalar 최종 shard 포맷은 dense-long 하나만 지원합니다.

## Runtime Dependencies

필요한 jar는 `java/lib`에 둡니다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

이 저장소의 Java package들은 thin jar입니다. package jar 안에 dependency를 묶지 않으므로 실행 시 필요한 jar를 classpath에 함께 넣어야 합니다.

공통 dependency:

- `duckdb_jdbc-1.1.3.jar`: metadata parquet, scalar dense-long parquet, array sample parquet read/write를 담당합니다.
- `jackson-core-2.20.0.jar`, `jackson-databind-2.20.0.jar`, `jackson-annotations-2.20.jar`: manifest/state JSON을 읽고 씁니다.

package별 추가 dependency:

- `array-binary-shard-java`: 공통 dependency만 필요합니다.
- `scalar-feature-shard-java`: 공통 dependency만 필요합니다.
- `array-sample-parquet-java`: 공통 dependency에 더해 Arrow bridge jar가 필요합니다. raw sample write에서 Java 배열을 Arrow vector batch로 묶어 DuckDB `registerArrowStream(...)`에 넘기기 위한 경로입니다.

현재 Java package 3개 기준으로 Hadoop/Parquet Java writer, Woodstox, stax2, commons jar는 필요하지 않습니다.

## Directory Structure

```text
java/
  src/
    fs/
      config/
      io/
        array/
        array_sample_parquet/
        common/
        scalar/
      model/
        array/
        array_sample_parquet/
        common/
        scalar/
        selection/
        synthetic/
      pipeline/
      synth/
      validate/
    scripts/
```

역할:

- `fs.io`: 사용자가 직접 여는 facade와 dataset entrypoint입니다.
- `fs.io.array`: array binary shard 구현입니다.
- `fs.io.array_sample_parquet`: array sample parquet 구현입니다.
- `fs.io.scalar`: scalar dense-long 하위 구현입니다.
- `fs.io.common`: DuckDB, JSON, dtype, array helper입니다.
- `fs.model.*`: public model과 manifest 구조입니다.
- `scripts`: local test, benchmark, example entrypoint입니다.

## Compile

```powershell
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" -encoding UTF-8 -cp "java\lib\*" -d java\out @java\sources.txt
```

소스와 javadoc에 한국어 주석이 있으므로 `-encoding UTF-8`을 항상 넣어야 합니다.

## Jar Build

Array binary:

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

Array sample parquet:

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_sample_parquet_java\build.ps1
```

Scalar:

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

## Config Quick Start

처음에는 아래 설정만 넣으면 됩니다.

```java
// scalar dense-long
BuildShardConfig scalarCfg = new BuildShardConfig();
scalarCfg.targetShardBytes = 32L * 1024L * 1024L;
scalarCfg.statsYCols = Arrays.asList("y");

// array sample parquet
ArraySampleParquetBuildOptions parquetOptions = new ArraySampleParquetBuildOptions();
parquetOptions.targetPartBytes = 128L * 1024L * 1024L;
parquetOptions.compression = "zstd";

// array custom binary
ArrayBinaryBuildOptions binaryOptions = new ArrayBinaryBuildOptions();
binaryOptions.samplesPerBlock = 16;
binaryOptions.targetShardMb = 32;
binaryOptions.codec = "none";
```

- `sampleKeyCol`, `featureKeyCol`은 metadata key column 이름이 기본값과 다를 때만 바꿉니다.
- scalar의 `denseLongRowGroupFeatures`는 기본 128을 권장합니다. 성능 테스트 후에만 조정하십시오.
- array sample parquet의 `arrowBatchRows`는 Java writer의 Arrow batch 크기입니다. 메모리 피크가 크면 줄이고, batch overhead가 크면 키웁니다.
- array custom binary는 빠른 serving용 특수 포맷입니다. 유지보수성이 중요하면 array sample parquet를 우선 사용하십시오.

## Array Public API

주요 entrypoint:

- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayDatasetBuilder`
- `fs.io.ArraySampleParquets`

### Array Binary Shard

`ArrayBinaryShards`는 custom binary array shard facade입니다.

- `open(manifestPath)`: shard reader를 엽니다.
- `loadManifest(...)`, `loadLocator(...)`, `loadSampleIds(...)`, `loadFeatureIds(...)`: lookup helper를 로드합니다.
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 작성합니다.
- `openSession(...)`: resumable array binary build session을 엽니다.

array binary builder는 sample context를 명시적으로 엽니다. sample 하나 안에 여러 feature trace가 들어가기 때문에 sample 경계를 public API로 드러내야 checkpoint가 안전합니다.

### Array Sample Parquet

`ArraySampleParquets`는 viewer/debugging용 sample-major Parquet facade입니다.

- builder는 sample별 raw parquet를 먼저 만듭니다.
- `finish()` 또는 `compact()`가 `sample_parts/`와 `trace_index_parts/`를 생성합니다.
- 최종 point row 정렬은 `(sample_id, feature_id, point_idx)`입니다.

## Scalar Public API

주요 entrypoint:

- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarDatasetBuilder`
- `fs.io.ScalarDenseLongDataset`

### Facade

`ScalarFeatureShards`가 scalar의 public entrypoint입니다.

- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 작성합니다.
- `openSession(...)`: sample별 raw parquet stage를 만드는 standard builder session을 열거나 재개합니다.
- `newBuilder(...)`: `openSession(...)`과 같은 builder를 생성하는 호환 entrypoint입니다.
- `buildDenseLongShardsFromSampleMajorManifest(...)`: sample-major manifest에서 dense-long shard를 만듭니다.
- `open(...)`, `openDenseLong(...)`: dense-long reader를 엽니다.
- `loadManifest(...)`: dense-long manifest를 읽습니다.

### Builder

별도 sequential builder나 raw builder는 없습니다. `ScalarDatasetBuilder` 하나가 표준 builder입니다.

- `status().pendingSampleIds`: 아직 commit되지 않은 sample id 목록입니다.
- `writeSample(sampleId, values, skipIfCompleted)`: sample 하나를 raw parquet로 commit합니다.
- `finishStage()`: raw sample 파일 목록을 `sample_major_manifest.json`으로 확정합니다.
- `buildShards(requireAll)`: 최종 dense-long shard를 생성합니다.
- `buildShards(requireAll, cleanupRaw)`: shard 생성 성공 후 `raw_samples/` parquet를 삭제할 수 있습니다.

순차 실행이 필요하면 `pendingSampleIds`를 앞에서부터 처리하면 됩니다. worker 병렬 실행이 필요하면 supervisor가 같은 목록을 나눠주면 됩니다.

### Dense-Long Shard

최종 scalar part parquet schema:

```text
feature_id  Int32
sample_id   Int64
mask        UInt8
value       Float64
```

모든 `(feature_id, sample_id)` 조합이 row로 존재합니다. missing은 내부 parquet에서 `mask=0, value=NaN`입니다. reader/API 사용자는 `mask=0`을 missing/null로 해석합니다. 물리 정렬은 `feature_id asc, sample_id asc`입니다.

기본 row group은 feature 128개 단위이며 `BuildShardConfig.denseLongRowGroupFeatures`로 조정할 수 있습니다.

## Scalar Example

```java
BuildShardConfig cfg = new BuildShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.statsYCols = Arrays.asList("y");

String manifestPath;
try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
        "C:\\data\\scalar_stage",
        "C:\\data\\sample_meta.parquet",
        "C:\\data\\feature_meta.parquet",
        null,
        cfg)) {
    for (Long sampleId : builder.status().pendingSampleIds) {
        builder.writeSample(sampleId.longValue(), row("feature_a", 1.23, "feature_b", null), true);
    }
    manifestPath = builder.buildShards(true);
}

try (ScalarDenseLongDataset ds = ScalarFeatureShards.open(manifestPath)) {
    ScalarFeatureValues values = ds.loadFeatureByKey("feature_a");
    System.out.println(values.values.size());
}
```

## Tests

전체 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunTestsMain --seed 0
```

주요 개별 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunArrayV3TestsMain
java -cp "java\lib\*;java\out" scripts.RunArraySampleParquetTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarNotebookBuilderTestsMain
```

## Script Inventory

현재 남긴 `java/src/scripts` 엔트리포인트는 다음 용도입니다.

- `BuildScalarDenseLongFromValueApiMain`, `BuildArraySampleParquetFromValueApiMain`: Python value API를 호출해서 Java builder로 전체 dataset을 만드는 예제입니다.
- `BuildShardsMain`, `BuildArrayShardsMain`: 이미 만들어진 sample-major manifest 또는 array bundle manifest에서 최종 shard를 만드는 CLI입니다.
- `GenerateScalarShardTestsMain`, `GenerateArrayShardTestsMain`, `GenerateArraySynthMain`: synthetic 데이터와 shard 생성 예제입니다.
- `Run*TestsMain`: 현재 유지하는 포맷별 검증 스크립트입니다.
- `LocateFeatureMain`, `LocateArrayFeatureMain`: feature가 어느 part/shard에 들어있는지 확인하는 디버깅 CLI입니다.
- `BenchmarkArraySampleParquetJavaMain`, `BenchmarkArrayReaderModesMain`: 유지 중인 array 경로의 성능 확인용 스크립트입니다.

## Reference

- array binary format: [docs/array_binary_shard_format_v3.md](../docs/array_binary_shard_format_v3.md)
- array sample parquet format: [docs/array_sample_parquet_format_v1.md](../docs/array_sample_parquet_format_v1.md)
- scalar dense-long format: [docs/scalar_parquet_shard_format.md](../docs/scalar_parquet_shard_format.md)
- scalar jar package: [packages/scalar_feature_shard_java/README.md](../packages/scalar_feature_shard_java/README.md)
