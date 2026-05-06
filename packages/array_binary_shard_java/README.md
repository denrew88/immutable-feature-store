Java용 array binary shard 라이브러리 패키지다.

이 패키지는 array binary shard v3의 reader, direct-ingestion builder, metadata helper, key resolver를 담는다.

진입점:
- `fs.io.ArrayBinaryShards`

## 준비물

- Java 8
- Java runtime dependency jars
  - `java/lib/duckdb_jdbc-1.1.3.jar`
  - `java/lib/jackson-core-2.20.0.jar`
  - `java/lib/jackson-databind-2.20.0.jar`
  - `java/lib/jackson-annotations-2.20.jar`

jar가 없으면 먼저 내려받는다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

## 빌드

```powershell
cd packages\array_binary_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

출력:
- `dist/array-binary-shard-java-0.3.0.jar`
- `dist/array-binary-shard-java-0.3.0-sources.jar`
- `dist/array-binary-shard-java-0.3.0-javadoc.jar`

thin jar이므로 실행 시 `java/lib/*` 아래의 DuckDB JDBC와 Jackson jar를 classpath에 같이 넣어야 한다.

IDE에서 Javadoc과 소스 탐색을 같이 쓰려면:
- binary jar: `array-binary-shard-java-0.3.0.jar`
- source attachment: `array-binary-shard-java-0.3.0-sources.jar`
- javadoc attachment: `array-binary-shard-java-0.3.0-javadoc.jar`

## Public API

### `fs.io.ArrayBinaryShards`

정적 facade다. 보통 외부 코드에서는 여기서 시작한다.

- `buildFromBundles(...)`
  - bundle manifest를 최종 array binary shard artifact로 변환한다.
- `loadManifest(...)`
  - `ArrayShardManifest`를 읽는다.
- `open(...)`
  - `ArrayBinaryShardReader`를 연다.
- `loadLocator(...)`
  - `feature_id -> shard/block 위치` 인덱스를 연다.
- `loadSampleIds(...)`
  - `sample_id <-> sample_key` 인덱스를 연다.
- `loadFeatureIds(...)`
  - `feature_id <-> feature_key` 인덱스를 연다.
- `writeSampleMeta(...)`
  - sample metadata parquet를 쓴다.
- `writeFeatureMeta(...)`
  - feature metadata parquet를 쓴다.
- `newBuilder(...)`
  - direct-ingestion builder를 만든다.

### `fs.io.ArrayBinaryShardReader`

binary shard를 실제로 읽는 저수준 reader다.

- `pointSchema()`
  - point schema를 돌려준다.
- `loadBlock(...)`
  - shard 안의 block payload 하나를 로드한다.
- `loadFeatureSamples(...)`
  - feature 하나와 sample id 배열을 받아 trace map을 만든다.
- `loadFeatureSamplesBySampleIds(...)`
  - sample id 인덱스를 포함한 helper 경로다.
- `loadFeatureSamplesBySampleKeys(...)`
  - sample key 기준으로 trace를 읽는다.
- `loadFeatureSamplesByKeys(...)`
  - feature key와 sample key 둘 다 외부 키 기준으로 조회한다.

categorical column은 `decodeCategorical=true`로 label까지 풀어서 읽을 수 있다.

### `fs.io.ArrayDatasetBuilder`

trace를 직접 넣어 bundle stage와 최종 shard를 만드는 builder다.

- `addTrace(sampleId, featureId, featureKey, columns)`
  - trace 하나를 추가한다.
- `sample(sampleId)`
  - sample context를 열고 그 안에서 `addTrace(...)`를 호출한다.
- `finishBundles()`
  - intermediate bundle stage를 확정한다.
- `updateFeatureMeta(records, on, requireAll)`
  - discovered-feature mode 뒤에 extra metadata를 merge한다.
- `buildShards([cleanupBundles])`
  - bundle stage를 최종 array binary shard로 변환한다.

## 사용 예제

```java
import fs.io.ArrayBinaryShards;
import fs.io.ArrayBinaryShardReader;
import fs.io.ArrayDatasetBuilder;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.model.array.ArrayTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.util.Arrays;
import java.util.Collections;
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

        try (ArrayDatasetBuilder builder = ArrayBinaryShards.newBuilder(
                "C:\\data\\array_shards",
                "C:\\data\\array_sample_meta.parquet",
                Arrays.asList(
                        new PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER, null),
                        new PointColumnSpec("state_code", StorageType.UINT32, LogicalType.CATEGORICAL, null)
                ),
                Collections.singletonList("feature_a"))) {
            builder.addTrace(
                    0L,
                    null,
                    "feature_a",
                    row("phase", new int[]{10, 11}, "state_code", new String[]{"OK", "WARN"})
            );
            builder.buildShards(true);
        }

        ArrayBinaryShardReader reader = ArrayBinaryShards.open("C:\\data\\array_shards\\array_binary_shard_manifest.json");
        ArrayFeatureLocatorIndex locator = ArrayBinaryShards.loadLocator(ArrayBinaryShards.loadManifest("C:\\data\\array_shards\\array_binary_shard_manifest.json"));
        Map<Long, ArrayTrace> traces = reader.loadFeatureSamples(0, new long[]{0L}, locator);
        System.out.println(traces.keySet());
        reader.close();
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

- dense id 규칙
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- manifest 기반 relative path를 사용한다.
- categorical column은 code로 저장하고 reader에서 label decode를 지원한다.
- 실제 포맷 설명은 `docs/array_binary_shard_format_v3.md`를 본다.
