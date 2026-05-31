# array-binary-shard-java

Java 8용 array custom binary shard v3 reader/builder thin jar입니다. 조회 속도는 빠르지만 포맷 유지보수 비용이 있으므로, viewer/debugging 목적이면 `array-sample-parquet-java`를 먼저 고려하십시오.

## 포함 API

- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayDatasetBuilder`
- `fs.config.ArrayBinaryBuildOptions`

지원 작업:

- dense metadata parquet 작성
- resumable array build session 실행
- 최종 array binary shard 생성
- 최종 array binary shard 조회

## 준비

필요 runtime dependency:

- `java/lib/duckdb_jdbc-1.1.3.jar`
- `java/lib/jackson-core-2.20.0.jar`
- `java/lib/jackson-databind-2.20.0.jar`
- `java/lib/jackson-annotations-2.20.jar`

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

산출물:

- `dist/array-binary-shard-java-0.3.0.jar`
- `dist/array-binary-shard-java-0.3.0-sources.jar`
- `dist/array-binary-shard-java-0.3.0-javadoc.jar`

thin jar이므로 실행 시 DuckDB JDBC와 Jackson jars를 classpath에 같이 넣어야 합니다. Hadoop/Parquet Java writer, Arrow, SLF4J, Woodstox, stax2, commons jar는 필요하지 않습니다.

## Config Guide

처음에는 아래 설정만 넣으면 됩니다.

```java
ArrayBinaryBuildOptions options = new ArrayBinaryBuildOptions();
options.samplesPerBlock = 16;
options.targetShardMb = 32;
options.codec = "none";
```

| option | 기본값 | 설명 |
| --- | --- | --- |
| `samplesPerBlock` | 16 | feature 하나를 sample 축으로 자르는 block 크기입니다. 작을수록 부분 조회 낭비가 줄지만 index overhead가 늘어납니다. |
| `targetShardMb` | 32 | shard 하나의 목표 크기입니다. shard가 너무 많으면 키우고, 한 shard가 너무 크면 줄입니다. |
| `nShards` | `null` | shard 개수를 직접 고정할 때만 넣습니다. 보통 자동 분할을 둡니다. |
| `codec` | `"none"` | payload codec입니다. 현재 Java custom binary 구현은 `"none"`만 실사용 경로로 봅니다. |
| `zstdLevel` | 3 | `codec="zstd"`일 때의 압축 level입니다. 현재 Java custom binary shard jar에는 zstd payload codec 구현이 없습니다. |
| `sampleKeyCol` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `featureKeyCol` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |

`ArrayBundleConfig`와 `ArrayShardConfig`는 bundle-to-shard 변환을 직접 호출할 때 쓰는 low-level 설정입니다. direct builder를 쓰는 일반 경로에서는 `ArrayBinaryBuildOptions`만 넣으면 됩니다.

## Public API

### `ArrayBinaryShards`

- `openSession(...)`: resumable array build session을 엽니다.
- `newBuilder(...)`: legacy alias입니다.
- `open(...)`: 최종 array binary shard reader를 엽니다.
- `loadManifest(...)`, `loadLocator(...)`, `loadSampleIds(...)`, `loadFeatureIds(...)`: low-level lookup helper를 로드합니다.
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 씁니다.

### `ArrayDatasetBuilder`

공통 lifecycle:

- `openSession(...)`
- `status()`
- `sample(sampleId)` 또는 `sample(sampleKey)`
- sample context 안에서 `addTrace(...)`
- `finishStage()`
- `buildShards(...)`

array 입력은 sample 하나 안에 trace 여러 개가 들어가므로 sample context가 resume-safe checkpoint 경계입니다.

### `ArrayBinaryShardReader`

- `loadFeatureSamples(...)`
- `loadFeatureSamplesBySampleKeys(...)`
- `loadFeatureSamplesByKeys(...)`

categorical column은 `decodeCategorical=true`로 label까지 복원할 수 있습니다.

## Example

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

## Reference

- 포맷 상세: [docs/array_binary_shard_format_v3.md](../../docs/array_binary_shard_format_v3.md)
- 전체 Java 사용법: [java/README.md](../../java/README.md)
