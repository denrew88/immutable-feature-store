# scalar-feature-shard-java

Java 8에서 scalar dense-long shard를 생성하고 조회하기 위한 jar 패키지입니다.

현재 scalar 최종 shard 포맷은 dense-long 하나만 지원합니다. 예전 feature-major scalar reader/builder API는 제거되었습니다.

## 준비

필요한 runtime jar는 `java/lib` 아래에 둡니다.

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

## 빌드

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

산출물:

- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-sources.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-javadoc.jar`

thin jar이므로 실행 시 `java/lib/*`를 classpath에 같이 넣어야 합니다.

## Public API

### `ScalarFeatureShards`

가장 바깥쪽 facade입니다.

- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 작성합니다.
- `openSession(...)`: sample 순서 기반 resumable build session을 엽니다.
- `openRawSession(...)`: sample별 raw parquet를 쓰는 random-order build session을 엽니다.
- `newBuilder(...)`: `openSession(...)`과 같은 builder를 만드는 호환 entrypoint입니다.
- `buildDenseLongShardsFromSampleBundles(...)`: sample-bundle/raw-sample manifest에서 dense-long shard를 만듭니다.
- `open(...)`, `openDenseLong(...)`: dense-long shard reader를 엽니다.
- `loadManifest(...)`: dense-long manifest를 읽습니다.

### `ScalarDatasetBuilder`

순차 builder입니다. sample은 `status().nextExpectedSampleId`부터 순서대로 써야 합니다.

Lifecycle:

- `openSession(...)`
- `status()`
- `writeSample(sampleId, values)`
- `finishStage()`
- `buildShards(requireAll)`

`buildShards(...)`는 항상 dense-long shard를 생성합니다.

### `ScalarRawDatasetBuilder`

random-order builder입니다. sample 하나를 raw parquet 하나로 commit하고, 나중에 dense-long shard로 묶습니다.

Lifecycle:

- `openRawSession(...)`
- `status()`
- `writeSample(sampleId, values, skipIfCompleted)`
- `finishStage()`
- `buildDenseLongShards(requireAll, outDir)`

`status().pendingSampleIds`로 아직 쓰지 않은 sample 목록을 확인할 수 있습니다.

### `ScalarDenseLongDataset`

dense-long shard reader입니다.

- `loadFeatureById(...)`, `loadFeatureByKey(...)`
- `loadSampleById(...)`, `loadSampleByKey(...)`
- `topFeaturesFromStats(...)`

## Dense-Long Format

최종 parquet row는 모든 `(feature_id, sample_id)` 조합을 담습니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8   # 1=present, 0=missing
value       Float64 # mask=0이면 무시
```

물리 정렬은 `feature_id asc, sample_id asc`입니다. 기본 row group은 feature 128개 단위이며 `BuildShardConfig.denseLongRowGroupFeatures`로 조정할 수 있습니다.

artifact 구조:

```text
scalar_dense_long_shard/
  dense_long_shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  dense_long_parts/
    part_0000.parquet
  selection_stats/
    y.parquet
```

## 예제

```java
import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarDenseLongDataset;
import fs.io.ScalarFeatureShards;
import fs.model.scalar.ScalarFeatureValues;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ScalarPackageExample {
    public static void main(String[] args) throws Exception {
        ScalarFeatureShards.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000", "y", 1.0),
                        row("sample_key", "sample_000001", "y", 2.0)
                ),
                "C:\\data\\sample_meta.parquet"
        );

        ScalarFeatureShards.writeFeatureMeta(
                Arrays.asList(
                        row("feature_key", "feature_a"),
                        row("feature_key", "feature_b")
                ),
                "C:\\data\\feature_meta.parquet"
        );

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
                if (sampleId == 0L) {
                    session.writeSample(sampleId, row("feature_a", 1.23, "feature_b", 4.56));
                } else {
                    session.writeSample(sampleId, row("feature_a", 7.89));
                }
            }

            manifestPath = session.buildShards(false);
        }

        try (ScalarDenseLongDataset ds = ScalarFeatureShards.open(manifestPath)) {
            ScalarFeatureValues values = ds.loadFeatureByKey("feature_a");
            System.out.println(values.values.size());
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

## Jar Example

jar classpath만으로 sample meta, feature meta, raw sample stage, dense-long scalar shard를 만드는 예제:

- `packages/scalar_feature_shard_java/examples/BuildScalarFeatureShardWithJarExample.java`

컴파일:

```powershell
New-Item -ItemType Directory -Force packages\scalar_feature_shard_java\examples\out | Out-Null
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  -d packages\scalar_feature_shard_java\examples\out `
  packages\scalar_feature_shard_java\examples\BuildScalarFeatureShardWithJarExample.java
```

실행:

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\scalar_feature_shard_java\examples\out;packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  BuildScalarFeatureShardWithJarExample
```

## Tests

```powershell
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarNotebookBuilderTestsMain
```

## 참고

- format 문서: [docs/scalar_parquet_shard_format.md](../../docs/scalar_parquet_shard_format.md)
- Java 전체 설명: [java/README.md](../../java/README.md)
