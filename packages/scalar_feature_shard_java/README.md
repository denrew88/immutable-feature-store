# scalar-feature-shard-java

Java 8용 scalar feature shard reader/builder/selection 패키지입니다.

## 포함 내용

- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarShardDataset`
- `fs.io.ScalarDatasetBuilder`
- `fs.io.ScalarRawDatasetBuilder`
- `fs.io.ScalarDenseLongDataset`

이 패키지는 다음 작업을 지원합니다.

- dense metadata parquet 작성
- resumable scalar build session 실행
- sample별 raw parquet를 임의 순서로 작성한 뒤 shard materialize
- 최종 scalar shard 읽기
- dense-long parquet shard 생성/읽기
- selection 후보 생성과 최종 선택

## 준비

- Java 8
- runtime dependency jars
  - `java/lib/duckdb_jdbc-1.1.3.jar`
  - `java/lib/jackson-core-2.20.0.jar`
  - `java/lib/jackson-databind-2.20.0.jar`
  - `java/lib/jackson-annotations-2.20.jar`
  - `java/lib/parquet-hadoop-bundle-1.13.1.jar`
  - `java/lib/hadoop-common-3.3.6.jar`
  - `java/lib/slf4j-api-1.7.36.jar`
  - `java/lib/woodstox-core-6.5.1.jar`
  - `java/lib/stax2-api-4.2.1.jar`
  - `java/lib/commons-collections-3.2.2.jar`
  - `java/lib/commons-lang3-3.12.0.jar`
  - `java/lib/hadoop-mapreduce-client-core-3.3.6.jar`

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

## 빌드

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

산출물:

- `dist/scalar-feature-shard-java-0.1.0.jar`
- `dist/scalar-feature-shard-java-0.1.0-sources.jar`
- `dist/scalar-feature-shard-java-0.1.0-javadoc.jar`

thin jar이므로 실행 시 DuckDB JDBC, Jackson, Parquet bundle, Hadoop common, Hadoop MapReduce core, SLF4J API, Woodstox, commons-collections, commons-lang3 jars를 classpath에 같이 넣어야 합니다.

## Public API

### `ScalarFeatureShards`

- `openSession(...)`
  - resumable scalar build session을 연다
- `openRawSession(...)`
  - sample별 raw parquet를 쓰는 random-order scalar session을 연다
- `newBuilder(...)`
  - legacy alias 역할을 한다
- `open(...)`
  - 기존 blob scalar shard reader를 연다
- `openDenseLong(...)`
  - dense-long scalar shard reader를 연다
- `buildDenseLongShardsFromSampleBundles(...)`
  - sample-bundle/raw-sample manifest에서 dense-long shard를 만든다
- `writeSampleMeta(...)`, `writeFeatureMeta(...)`
  - dense metadata parquet를 쓴다
- `buildCandidates(...)`
  - selection 후보만 만든다
- `selectFeatures(...)`
  - 후보를 만든 뒤 최종 선택까지 진행한다

### `ScalarDatasetBuilder`

공통 lifecycle:

- `openSession(...)`
- `status()`
- `writeSample(sampleId, values)`
- `finishStage()`
- `buildShards(...)`

중요:

- scalar public write 단위는 sample 하나입니다.
- `writeValue(...)` 같은 per-value public path는 제공하지 않습니다.
- resume는 `status().nextExpectedSampleId`를 기준으로 합니다.

### `ScalarRawDatasetBuilder`

Python `ScalarRawDatasetBuilder`와 같은 random-order ingest 모델입니다.

공통 lifecycle:

- `ScalarFeatureShards.openRawSession(...)`
- `status()`
- `writeSample(sampleId, values, skipIfCompleted)`
- `finishStage()`
- `buildBlobShards(...)` 또는 `buildDenseLongShards(...)`

중요:

- sample 하나가 `raw_samples/sample_*.parquet` 파일 하나로 commit됩니다.
- sample은 순서와 무관하게 작성할 수 있습니다.
- 완료 여부는 `raw_samples.jsonl` commit log 기준입니다.
- `status().pendingSampleIds`를 worker에게 나눠주면 중단 후 재개와 외부 병렬 처리를 같은 방식으로 다룰 수 있습니다.

### `ScalarShardDataset`

- `getValue(...)`, `getValueByKey(...)`
- `getValues(...)`, `getValuesByKeys(...)`
- `iterMany(...)`, `iterManyByKey(...)`

### `ScalarDenseLongDataset`

dense-long shard는 모든 `(feature_id, sample_id)` 조합을 parquet row로 저장합니다.

```text
feature_id  Int32
sample_id   Int64
mask        UInt8   # 1=present, 0=missing
value       Float64 # mask=0이면 무시
```

기본 row group은 feature 128개 단위입니다. feature 하나 조회는 해당 feature가 들어 있는 row group을 읽고, sample 기준 조회는 여러 part에 `sample_id` filter를 걸어 표준 parquet scan으로 처리합니다.

주요 API:

- `loadFeatureById(...)`, `loadFeatureByKey(...)`
- `loadSampleById(...)`, `loadSampleByKey(...)`
- `topFeaturesFromStats(...)`

## 예제

```java
import fs.config.BuildShardConfig;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;
import fs.io.ScalarShardDataset;
import fs.model.scalar.ScalarFeatureValues;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ScalarPackageExample {
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

        try (ScalarShardDataset ds = ScalarFeatureShards.open("C:\\data\\scalar_shards\\shard_manifest.json")) {
            ScalarFeatureValues values = ds.getValuesByKeys(
                    "feature_a",
                    new String[]{"sample_000000", "sample_000001"}
            );
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

### Raw + Dense-Long 예제

```java
BuildShardConfig cfg = new BuildShardConfig();
cfg.targetShardBytes = 32L * 1024L * 1024L;
cfg.statsYCols = Arrays.asList("y");
cfg.denseLongRowGroupFeatures = 128;

try (ScalarRawDatasetBuilder session = ScalarFeatureShards.openRawSession(
        "C:\\data\\scalar_raw",
        "C:\\data\\sample_meta.parquet",
        "C:\\data\\feature_meta.parquet",
        null,
        cfg)) {
    session.writeSample(10L, row("feature_a", 1.23), true);
    session.writeSample(3L, row("feature_a", 2.34, "feature_b", 5.67), true);
    System.out.println(session.status().pendingSampleIds.size());
}

try (ScalarRawDatasetBuilder session = ScalarFeatureShards.openRawSession(
        "C:\\data\\scalar_raw",
        "C:\\data\\sample_meta.parquet",
        "C:\\data\\feature_meta.parquet",
        null,
        cfg)) {
    String denseManifest = session.buildDenseLongShards(true, "C:\\data\\scalar_dense_long");

    try (ScalarDenseLongDataset ds = ScalarFeatureShards.openDenseLong(denseManifest)) {
        ScalarFeatureValues values = ds.loadFeatureByKey("feature_a");
        System.out.println(values.values.size());
    }
}
```

## Jar Example

sample meta, feature meta, raw sample stage, dense-long scalar shard를 jar classpath만으로 생성하는 전체 예제는 다음 파일에 있습니다.

- `examples/BuildScalarFeatureShardWithJarExample.java`

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

기본 출력 위치는 `data/tmp_scalar_feature_shard_jar_example`입니다. 다른 위치에 쓰려면 실행 명령 끝에 출력 root directory를 인자로 넘기면 됩니다.

## 참고

- 포맷 상세: [docs/scalar_parquet_shard_format.md](../../docs/scalar_parquet_shard_format.md)
- 전체 Java 사용법: [java/README.md](../../java/README.md)
