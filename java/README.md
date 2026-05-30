# Java 구현 안내

이 디렉터리는 Java 8 기준의 array/scalar shard 구현, 테스트 스크립트, 그리고 jar 패키지 빌드 스크립트를 담고 있습니다.

핵심 원칙은 다음과 같습니다.

- array/scalar 모두 dense `sample_id`, `feature_id`를 기준으로 동작합니다.
- direct-ingestion builder는 **resumable session** 모델을 사용합니다.
- 자동 chunking은 내부 구현으로 숨기고, 사용자는 `status()`의 sample watermark만 보고 이어서 넣으면 됩니다.

## 준비

- Java 8
- runtime dependency jars
  - `java/lib/duckdb_jdbc-1.1.3.jar`
  - `java/lib/jackson-core-2.20.0.jar`
  - `java/lib/jackson-databind-2.20.0.jar`
  - `java/lib/jackson-annotations-2.20.jar`
  - `java/lib/parquet-hadoop-bundle-1.13.1.jar`
  - `java/lib/hadoop-common-3.3.6.jar`
  - `java/lib/hadoop-mapreduce-client-core-3.3.6.jar`
  - `java/lib/slf4j-api-1.7.36.jar`
  - `java/lib/woodstox-core-6.5.1.jar`
  - `java/lib/stax2-api-4.2.1.jar`
  - `java/lib/commons-collections-3.2.2.jar`
  - `java/lib/commons-lang3-3.12.0.jar`
  - `java/lib/arrow-c-data-14.0.2.jar`
  - `java/lib/arrow-memory-core-14.0.2.jar`
  - `java/lib/arrow-memory-unsafe-14.0.2.jar`
  - `java/lib/arrow-vector-14.0.2-shade-format-flatbuffers.jar`
  - `java/lib/netty-common-4.1.96.Final.jar`

없으면 먼저 받습니다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

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

역할 분리는 이렇게 보면 됩니다.

- `fs.io`
  - 외부 사용자가 직접 여는 facade와 dataset entrypoint
- `fs.io.array`
  - array bundle writer, shard builder, manifest/idx/bin reader
- `fs.io.scalar`
  - scalar sample-bundle writer, shard builder, parquet reader
- `fs.io.common`
  - DuckDB, JSON, dtype/array helper
- `fs.model.*`
  - 메모리 안에서의 데이터 구조 정의

## 컴파일

저장소 루트에서 실행합니다.

```powershell
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" -encoding UTF-8 -cp "java\lib\*" -d java\out @java\sources.txt
```

Javadoc이 한글이므로 `-encoding UTF-8`은 항상 넣는 것이 좋습니다.

## jar 빌드

### Array

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

산출물:

- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0.jar`
- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0-sources.jar`
- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0-javadoc.jar`

### Scalar

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

산출물:

- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-sources.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-javadoc.jar`

두 jar 모두 thin jar입니다. 실행 시에는 DuckDB JDBC, Jackson, Parquet bundle, Hadoop common, Hadoop MapReduce core, SLF4J API, Woodstox, commons-collections, commons-lang3 jars를 classpath에 같이 넣어야 합니다.

## Array Public API

진입점:

- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayDatasetBuilder`

### Facade

`ArrayBinaryShards`가 가장 바깥 entrypoint입니다.

- `open(manifestPath)`
  - 최종 array binary shard를 reader로 엽니다.
- `loadManifest(...)`, `loadLocator(...)`, `loadSampleIds(...)`, `loadFeatureIds(...)`
  - low-level lookup helper를 로드합니다.
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`
  - dense metadata parquet를 씁니다.
- `openSession(...)`
  - resumable array build session을 엽니다.
- `newBuilder(...)`
  - 현재도 동작하지만, 새 코드는 `openSession(...)`을 권장합니다.

### Resumable session

array builder는 trace 입력을 받되, durability 기준은 trace 하나가 아니라 **committed bundle parquet 파일 하나**입니다.

공통 lifecycle:

- `openSession(...)`
- `status()`
- `sample(sampleId)` 또는 `sample(sampleKey)`
- sample context 안에서 `addTrace(...)`
- `finishStage()`
- `buildShards(...)`

`sample(...)`가 필요한 이유:

- array는 sample 하나 안에 feature trace 여러 개가 들어가는 구조입니다.
- builder는 sample 경계를 알아야 다른 sample trace가 섞이는 실수를 막을 수 있습니다.
- 자동 checkpoint commit도 sample이 완전히 닫힌 뒤에만 안전하게 판단할 수 있습니다.
- 그래서 array 쪽 sample context는 단순 편의 문법이 아니라, resume-safe ingestion 경계를 표시하는 public API입니다.

`status()`에서 중요한 값:

- `lastCommittedSampleId`
- `nextExpectedSampleId`

재시작 시에는 `nextExpectedSampleId`부터 다시 넣으면 됩니다.

### Array session 예제

```java
import fs.config.ArrayBinaryBuildOptions;
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ArraySessionExample {
    public static void main(String[] args) throws Exception {
        ArrayBinaryShards.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000", "split", "train"),
                        row("sample_key", "sample_000001", "split", "test")
                ),
                "C:\\data\\array_sample_meta.parquet"
        );

        ArrayBinaryShards.writeFeatureMeta(
                Arrays.asList(
                        row("feature_key", "feature_a", "group", "alpha")
                ),
                "C:\\data\\array_feature_meta.parquet"
        );

        ArrayBinaryBuildOptions options = new ArrayBinaryBuildOptions();
        options.samplesPerBlock = 16;
        options.targetShardMb = 32;
        options.codec = "none";

        try (ArrayDatasetBuilder session = ArrayBinaryShards.openSession(
                "C:\\data\\array_shards",
                "C:\\data\\array_sample_meta.parquet",
                Arrays.asList(
                        new PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
                        new PointColumnSpec("state_code", StorageType.UINT32, LogicalType.CATEGORICAL)
                ),
                "C:\\data\\array_feature_meta.parquet",
                options)) {

            long next = session.status().nextExpectedSampleId;
            for (long sampleId = next; sampleId < 2; sampleId++) {
                try (ArrayDatasetBuilder.ArraySampleContext sample = session.sample(sampleId)) {
                    sample.addTrace(
                            null,
                            "feature_a",
                            row(
                                    "phase", new int[]{10, 11, 12},
                                    "state_code", new String[]{"OK", "OK", "WARN"}
                            )
                    );
                }
            }

            session.finishStage();
            session.buildShards(false);
        }
    }

    private static Map<String, Object> row(Object... kv) {
        Map<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put((String) kv[i], kv[i + 1]);
        }
        return out;
    }
}
```

### Reader 예제

```java
import fs.io.ArrayBinaryShardReader;
import fs.io.ArrayBinaryShards;
import fs.io.array.ArrayFeatureIdIndex;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArraySampleIdIndex;
import fs.model.array.ArrayTrace;

import java.util.Map;

public class ArrayReadExample {
    public static void main(String[] args) throws Exception {
        String manifestPath = "C:\\data\\array_shards\\array_binary_shard_manifest.json";
        ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(ArrayBinaryShards.loadManifest(manifestPath));
        ArrayFeatureIdIndex featureIds = ArrayBinaryShards.loadFeatureIds(ArrayBinaryShards.loadManifest(manifestPath));
        ArraySampleIdIndex sampleIds = ArrayBinaryShards.loadSampleIds(ArrayBinaryShards.loadManifest(manifestPath));

        try (ArrayBinaryShardReader reader = ArrayBinaryShards.open(manifestPath)) {
            Map<String, ArrayTrace> traces = reader.loadFeatureSamplesByKeys(
                    "feature_a",
                    new String[]{"sample_000000"},
                    locator,
                    featureIds,
                    sampleIds,
                    true
            );
            System.out.println(traces.get("sample_000000").columns);
        }
    }
}
```

## Scalar Public API

진입점:

- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarShardDataset`
- `fs.io.ScalarDatasetBuilder`

### Facade

`ScalarFeatureShards`가 가장 바깥 entrypoint입니다.

- `open(manifestPath)`
  - 최종 scalar shard dataset을 reader로 엽니다.
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`
  - dense metadata parquet를 씁니다.
- `openSession(...)`
  - resumable scalar build session을 엽니다.
- `newBuilder(...)`
  - 현재도 동작하지만, 새 코드는 `openSession(...)`을 권장합니다.
- `buildCandidates(...)`
  - selection 후보만 만듭니다.
- `selectFeatures(...)`
  - 후보를 만든 뒤 최종 선택까지 진행합니다.

### Resumable session

scalar session은 public write 단위를 **sample 하나**로 고정합니다.

공통 lifecycle:

- `openSession(...)`
- `status()`
- `writeSample(sampleId, values)`
- `finishStage()`
- `buildShards(...)`

중요:

- `writeValue(...)` 같은 per-value public path는 제공하지 않습니다.
- sample은 반드시 `status().nextExpectedSampleId`부터 순서대로 넣어야 합니다.

### Scalar session 예제

```java
import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ScalarSessionExample {
    public static void main(String[] args) throws Exception {
        ScalarFeatureShards.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000", "y", 1.0, "y_alt", 1.5),
                        row("sample_key", "sample_000001", "y", 2.0, "y_alt", 2.5)
                ),
                "C:\\data\\sample_meta.parquet"
        );

        ScalarFeatureShards.writeFeatureMeta(
                Arrays.asList(
                        row("feature_key", "feature_a", "group", "alpha"),
                        row("feature_key", "feature_b", "group", "beta")
                ),
                "C:\\data\\feature_meta.parquet"
        );

        BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 32L * 1024L * 1024L;
        cfg.statsYCols = Arrays.asList("y", "y_alt");

        try (ScalarDatasetBuilder session = ScalarFeatureShards.openSession(
                "C:\\data\\scalar_shards",
                "C:\\data\\sample_meta.parquet",
                "C:\\data\\feature_meta.parquet",
                null,
                cfg,
                "C:\\data\\scalar_stage")) {

            long next = session.status().nextExpectedSampleId;
            for (long sampleId = next; sampleId < 2; sampleId++) {
                if (sampleId == 0L) {
                    session.writeSample(sampleId, row("feature_a", 1.23, "feature_b", 4.56));
                } else {
                    session.writeSample(sampleId, row("feature_a", 7.89));
                }
            }

            session.finishStage();
            session.buildShards(false);
        }
    }

    private static Map<String, Object> row(Object... kv) {
        Map<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put((String) kv[i], kv[i + 1]);
        }
        return out;
    }
}
```

### Reader / selection 예제

```java
import fs.config.SelectionConfig;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarShardDataset;
import fs.model.scalar.ScalarFeatureValues;

public class ScalarReadExample {
    public static void main(String[] args) throws Exception {
        String manifestPath = "C:\\data\\scalar_shards\\shard_manifest.json";

        try (ScalarShardDataset ds = ScalarFeatureShards.open(manifestPath)) {
            ScalarFeatureValues values = ds.getValuesByKeys(
                    "feature_a",
                    new String[]{"sample_000000", "sample_000001"}
            );
            System.out.println(values.values.size());
        }

        SelectionConfig cfg = new SelectionConfig();
        cfg.topM = 100;
        System.out.println(ScalarFeatureShards.selectFeatures(manifestPath, "y", cfg).size());
    }
}
```

## 테스트

전체 Java 테스트:

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

- array 포맷 상세: [docs/array_binary_shard_format_v3.md](../docs/array_binary_shard_format_v3.md)
- scalar 포맷 상세: [docs/scalar_parquet_shard_format.md](../docs/scalar_parquet_shard_format.md)
- package별 사용법:
  - [packages/array_binary_shard_java/README.md](../packages/array_binary_shard_java/README.md)
  - [packages/scalar_feature_shard_java/README.md](../packages/scalar_feature_shard_java/README.md)

## Array Sample Parquet

`fs.io.ArraySampleParquets`는 viewer/debugging용 sample-major Parquet 포맷의 public facade입니다.

- `openSession(...)`으로 sample 순서 ingest를 시작합니다.
- `status().nextExpectedSampleId`로 resume 위치를 확인합니다.
- `sample(sampleId)` / `sample(sampleKey)` context 안에서 trace를 추가합니다.
- sample close 시 trace 목록을 `(sample_id, feature_id)` 순서로 정렬한 뒤 Arrow vector batch를 DuckDB에 넘겨 `.parquet.tmp` raw 파일을 씁니다.
- raw write와 compact 단계는 이미 정렬된 입력을 사용하므로 SQL `ORDER BY`를 수행하지 않습니다.
- `ArraySampleParquetOrderChecks`로 raw/final parquet의 물리 row 정렬을 검증할 수 있습니다.
- part 크기는 sample 개수가 아니라 `targetPartBytes` 기준으로 자동 조절합니다.
- `finish()`가 `array_sample_parquet_manifest.json`을 씁니다.
- `ArraySampleParquetReader`는 `loadTracesByIds(...)`, `loadTracesByKeys(...)`를 제공합니다.

```java
ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
options.targetPartBytes = 128L * 1024L * 1024L;
options.compression = "none";

try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
        "C:\\data\\array_sample_parquet",
        "C:\\data\\sample_meta.parquet",
        pointSchema,
        "C:\\data\\feature_meta.parquet",
        options)) {
    long start = builder.status().nextExpectedSampleId;
    for (long sampleId = start; sampleId < nSamples; sampleId++) {
        try (ArraySampleParquetSampleContext sample = builder.sample(sampleId)) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
    builder.finish();
}
```

추가 테스트:

```powershell
java -cp "java\lib\*;java\out" scripts.RunArraySampleParquetTestsMain
```

추가 문서와 패키지:

- [docs/array_sample_parquet_format_v1.md](../docs/array_sample_parquet_format_v1.md)
- [packages/array_sample_parquet_java/README.md](../packages/array_sample_parquet_java/README.md)
