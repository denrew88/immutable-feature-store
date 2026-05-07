# array-binary-shard-java

Java 8용 array binary shard v3 reader/builder 패키지입니다.

## 포함 내용

- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayDatasetBuilder`

이 패키지는 다음 작업을 지원합니다.

- dense metadata parquet 작성
- resumable array build session 실행
- 최종 array binary shard 읽기

## 준비

- Java 8
- runtime dependency jars
  - `java/lib/duckdb_jdbc-1.1.3.jar`
  - `java/lib/jackson-core-2.20.0.jar`
  - `java/lib/jackson-databind-2.20.0.jar`
  - `java/lib/jackson-annotations-2.20.jar`

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

## 빌드

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

산출물:

- `dist/array-binary-shard-java-0.3.0.jar`
- `dist/array-binary-shard-java-0.3.0-sources.jar`
- `dist/array-binary-shard-java-0.3.0-javadoc.jar`

thin jar이므로 실행 시 DuckDB JDBC와 Jackson jars를 classpath에 같이 넣어야 합니다.

## Public API

### `ArrayBinaryShards`

- `openSession(...)`
  - resumable array build session을 연다
- `newBuilder(...)`
  - legacy alias 역할을 한다
- `open(...)`
  - 최종 array binary shard reader를 연다
- `loadManifest(...)`, `loadLocator(...)`, `loadSampleIds(...)`, `loadFeatureIds(...)`
  - low-level lookup helper를 로드한다
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`
  - dense metadata parquet를 쓴다

### `ArrayDatasetBuilder`

공통 lifecycle:

- `openSession(...)`
- `status()`
- `sample(sampleId)` 또는 `sample(sampleKey)`
- sample context 안에서 `addTrace(...)`
- `finishStage()`
- `buildShards(...)`

중요:

- 자동 chunking은 내부 구현입니다.
- 커밋 단위는 bundle parquet 하나입니다.
- resume는 `status().nextExpectedSampleId`를 기준으로 합니다.

### `ArrayBinaryShardReader`

- `loadFeatureSamples(...)`
- `loadFeatureSamplesBySampleKeys(...)`
- `loadFeatureSamplesByKeys(...)`

categorical column은 `decodeCategorical=true`로 label까지 복원할 수 있습니다.

## 예제

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

public class ArrayPackageExample {
    public static void main(String[] args) throws Exception {
        ArrayBinaryShards.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000"),
                        row("sample_key", "sample_000001")
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
                                    "state_code", new String[]{"OK", "WARN", "WARN"}
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

## 참고

- 포맷 상세: [docs/array_binary_shard_format_v3.md](../../docs/array_binary_shard_format_v3.md)
- 전체 Java 사용법: [java/README.md](../../java/README.md)
