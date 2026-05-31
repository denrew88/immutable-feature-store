# Java 구현 안내

이 디렉터리는 Java 8 기준 array/scalar builder, reader, test script, jar package build script를 담고 있습니다.

공통 원칙:

- array와 scalar 모두 dense `sample_id`, `feature_id`를 물리 id로 사용합니다.
- build API는 중단 후 재개 가능한 session 모델을 사용합니다.
- scalar 최종 shard 포맷은 dense-long 하나만 지원합니다.

## 준비

필요한 jar는 `java/lib`에 둡니다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

주요 dependency:

- DuckDB JDBC
- Jackson
- parquet-hadoop-bundle
- Hadoop common/mapreduce core
- SLF4J API
- Woodstox
- commons-collections, commons-lang3

## 디렉터리 구조

```text
java/
  src/
    fs/
      config/
      io/
        array/
        common/
        scalar/
      model/
        array/
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
- `fs.io.array`: array binary shard와 array sample parquet 구현입니다.
- `fs.io.scalar`: scalar dense-long builder/reader 구현입니다.
- `fs.io.common`: DuckDB, JSON, dtype, array helper입니다.
- `fs.model.*`: public model과 manifest 구조입니다.
- `scripts`: local test, benchmark, example entrypoint입니다.

## 컴파일

```powershell
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" -encoding UTF-8 -cp "java\lib\*" -d java\out @java\sources.txt
```

소스와 javadoc에 한글 주석이 있으므로 `-encoding UTF-8`을 항상 넣습니다.

## Jar 빌드

Array:

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

Scalar:

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

jar는 thin jar입니다. 실행할 때 `java/lib/*` dependency를 classpath에 같이 넣어야 합니다.

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
- `openSession(...)`: resumable array build session을 엽니다.

array builder는 sample context를 명시적으로 엽니다. array trace는 sample 하나 안에 여러 feature trace가 들어갈 수 있으므로 sample 경계를 public API로 드러내야 안전하게 checkpoint할 수 있습니다.

```java
try (ArrayDatasetBuilder session = ArrayBinaryShards.openSession(
        outDir,
        sampleMetaPath,
        pointSchema,
        featureMetaPath,
        options)) {
    try (ArrayDatasetBuilder.ArraySampleContext sample = session.sample(0L)) {
        sample.addTrace(null, "feature_a", columns);
    }
    session.finishStage();
    session.buildShards(false);
}
```

### Array Sample Parquet

`ArraySampleParquets`는 viewer/debugging용 sample-major Parquet facade입니다.

- raw builder는 sample별 raw parquet를 만들 수 있습니다.
- compact 단계에서 `sample_parts/`와 `trace_index_parts/`로 묶습니다.
- 최종 row 정렬은 `(sample_id, feature_id, point_idx)`입니다.

## Scalar Public API

주요 entrypoint:

- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarDatasetBuilder`
- `fs.io.ScalarRawDatasetBuilder`
- `fs.io.ScalarDenseLongDataset`

### Facade

`ScalarFeatureShards`가 scalar의 바깥 entrypoint입니다.

- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 작성합니다.
- `openSession(...)`: 순차 resumable build session을 엽니다.
- `openRawSession(...)`: random-order raw sample session을 엽니다.
- `newBuilder(...)`: `openSession(...)`과 같은 builder를 생성하는 호환 entrypoint입니다.
- `buildDenseLongShardsFromSampleBundles(...)`: sample-bundle/raw-sample manifest에서 dense-long shard를 만듭니다.
- `open(...)`, `openDenseLong(...)`: dense-long reader를 엽니다.
- `loadManifest(...)`: dense-long manifest를 읽습니다.

### Sequential Session

순차 session은 sample 단위로 씁니다.

- `openSession(...)`
- `status()`
- `writeSample(sampleId, values)`
- `finishStage()`
- `buildShards(requireAll)`

`writeSample(...)`은 반드시 `status().nextExpectedSampleId`부터 순서대로 호출해야 합니다. `buildShards(...)`는 항상 dense-long shard를 생성합니다.

### Random Raw Session

raw session은 sample 하나를 `raw_samples/sample_*.parquet` 파일 하나로 commit합니다. sample 순서 제약이 없습니다.

- `openRawSession(...)`
- `status()`
- `writeSample(sampleId, values, skipIfCompleted)`
- `finishStage()`
- `buildDenseLongShards(requireAll, outDir)`

`status().pendingSampleIds`는 아직 commit되지 않은 sample id 목록입니다. 외부 supervisor가 이 목록을 worker에게 나눠주면 중단 후 재개와 병렬 생성이 단순해집니다.

### Dense-Long Shard

최종 scalar part parquet schema:

```text
feature_id  Int32
sample_id   Int64
mask        UInt8
value       Float64
```

모든 `(feature_id, sample_id)` 조합이 row로 존재합니다. missing은 `mask=0`입니다. 물리 정렬은 `feature_id asc, sample_id asc`입니다.

기본 row group은 feature 128개 단위이며 `BuildShardConfig.denseLongRowGroupFeatures`로 조정할 수 있습니다.

### Scalar 예제

```java
BuildShardConfig cfg = new BuildShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.statsYCols = Arrays.asList("y");

String manifestPath;
try (ScalarDatasetBuilder session = ScalarFeatureShards.openSession(
        "C:\\data\\scalar_dense_long",
        "C:\\data\\sample_meta.parquet",
        "C:\\data\\feature_meta.parquet",
        null,
        cfg,
        "C:\\data\\scalar_stage")) {
    long next = session.status().nextExpectedSampleId;
    for (long sampleId = next; sampleId < 2; sampleId++) {
        session.writeSample(sampleId, row("feature_a", 1.23, "feature_b", 4.56));
    }
    manifestPath = session.buildShards(false);
}

try (ScalarDenseLongDataset ds = ScalarFeatureShards.open(manifestPath)) {
    ScalarFeatureValues values = ds.loadFeatureByKey("feature_a");
    System.out.println(values.values.size());
}
```

## 테스트

전체 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunTestsMain --seed 0
```

주요 개별 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArrayBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunArrayV3TestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarNotebookBuilderTestsMain
```

## 참고

- array binary format: [docs/array_binary_shard_format_v3.md](../docs/array_binary_shard_format_v3.md)
- array sample parquet format: [docs/array_sample_parquet_format_v1.md](../docs/array_sample_parquet_format_v1.md)
- scalar dense-long format: [docs/scalar_parquet_shard_format.md](../docs/scalar_parquet_shard_format.md)
- scalar jar package: [packages/scalar_feature_shard_java/README.md](../packages/scalar_feature_shard_java/README.md)
