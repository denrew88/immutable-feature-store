# scalar-feature-shard-java

Java 8에서 scalar dense-long shard를 생성하고 조회하기 위한 thin jar 패키지입니다. 현재 scalar 최종 shard 포맷은 dense-long 하나만 지원합니다.

## Runtime Dependencies

필요한 runtime jar는 `java/lib`에 둡니다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
```

`scalar-feature-shard-java` package jar는 dependency를 내부에 묶지 않는 thin jar입니다. 실행 시 다음 jar를 classpath에 함께 넣어야 합니다.

- `duckdb_jdbc-1.1.3.jar`
- `jackson-core-2.20.0.jar`
- `jackson-databind-2.20.0.jar`
- `jackson-annotations-2.20.jar`

scalar parquet write/read와 zstd 압축은 DuckDB JDBC가 담당합니다. JSON manifest/state 처리는 Jackson이 담당합니다. Hadoop/Parquet Java writer, Arrow, SLF4J, Woodstox, stax2, commons jar는 현재 scalar 구현에 필요하지 않습니다.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

출력물:

- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-sources.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-javadoc.jar`

## Public API

### `ScalarFeatureShards`

사용자가 주로 쓰는 facade입니다.

- `writeSampleMeta(...)`, `writeFeatureMeta(...)`: dense metadata parquet를 작성합니다.
- `openSession(...)`: sample별 raw parquet stage를 만드는 standard builder session을 열거나 재개합니다.
- `newBuilder(...)`: `openSession(...)`과 같은 builder를 생성하는 호환 entrypoint입니다.
- `buildDenseLongShardsFromSampleMajorManifest(...)`: sample-major manifest에서 dense-long shard를 만듭니다.
- `open(...)`, `openDenseLong(...)`: dense-long reader를 엽니다.
- `loadManifest(...)`: dense-long manifest를 읽습니다.

### `ScalarDatasetBuilder`

표준 builder입니다. sample id 순서를 강제하지 않고, 완료된 sample 하나를 `raw_samples/sample_*.parquet` 파일 하나로 commit합니다.

Lifecycle:

- `openSession(...)`
- `status()`
- `writeSample(sampleId, values, skipIfCompleted)`
- `finishStage()`
- `buildShards(requireAll)`
- `buildDenseLongShards(requireAll, outDir)`

`status().pendingSampleIds`로 아직 완료되지 않은 sample 목록을 확인할 수 있습니다. 순차 실행이 필요하면 이 목록을 앞에서부터 처리하면 됩니다.

### `ScalarDenseLongDataset`

dense-long shard reader입니다.

- `loadFeatureById(...)`, `loadFeatureByKey(...)`
- `loadSampleById(...)`, `loadSampleByKey(...)`
- `topFeaturesFromStats(...)`

## Example

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
                Arrays.asList(row("feature_key", "feature_a"), row("feature_key", "feature_b")),
                "C:\\data\\feature_meta.parquet"
        );

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
                Map<Object, Object> values = new LinkedHashMap<Object, Object>();
                values.put("feature_a", sampleId.doubleValue());
                values.put("feature_b", null);
                builder.writeSample(sampleId.longValue(), values, true);
            }
            manifestPath = builder.buildShards(true);
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
- `packages/scalar_feature_shard_java/examples/BuildScalarDenseLongFromValueApiWithJarExample.java`: 이미 존재하는 sample/feature meta를 기준으로 Python value API를 호출해서 dense-long shard를 생성합니다.

컴파일:

```powershell
New-Item -ItemType Directory -Force packages\scalar_feature_shard_java\examples\out | Out-Null
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  -d packages\scalar_feature_shard_java\examples\out `
  packages\scalar_feature_shard_java\examples\BuildScalarFeatureShardWithJarExample.java
```

Python value API 호출 예제를 컴파일하려면 마지막 파일명만 바꾸면 됩니다.

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  -d packages\scalar_feature_shard_java\examples\out `
  packages\scalar_feature_shard_java\examples\BuildScalarDenseLongFromValueApiWithJarExample.java
```

실행:

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\scalar_feature_shard_java\examples\out;packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  BuildScalarFeatureShardWithJarExample
```

Python value API 호출 예제는 `python/scripts/serve_synthetic_value_api.py`가 먼저 떠 있어야 하며, sample meta와 feature meta는 이미 생성되어 있어야 합니다.

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\scalar_feature_shard_java\examples\out;packages\scalar_feature_shard_java\dist\scalar-feature-shard-java-0.1.0.jar;java\lib\*" `
  BuildScalarDenseLongFromValueApiWithJarExample `
  --base-url http://127.0.0.1:8010 `
  --sample-meta data\sample_meta.parquet `
  --feature-meta data\feature_meta.parquet `
  --out-dir data\scalar_dense_long_from_api
```

API 서버 실행:

```powershell
python python\scripts\serve_synthetic_value_api.py --host 127.0.0.1 --port 8010
```

`BuildScalarDenseLongFromValueApiWithJarExample`는 sample 하나와 feature id 묶음마다 `POST /scalar/values`를 호출합니다. 요청은 `sample_id` 또는 `sample_key` 중 하나, `feature_ids` 또는 `feature_keys` 중 하나만 허용합니다.

요청 예:

```json
{
  "sample_meta_path": "data/sample_meta.parquet",
  "feature_meta_path": "data/feature_meta.parquet",
  "sample_id": 0,
  "feature_ids": [0, 1, 2],
  "seed": 7,
  "missing_rate": 0.1,
  "n_latent_groups": 16,
  "noise_scale": 0.25
}
```

응답 예:

```json
{
  "sample_id": 0,
  "sample_key": "sample_000000",
  "feature_count": 3,
  "values": [
    {
      "feature_id": 0,
      "feature_key": "feature_000000",
      "present": true,
      "value": 1.23
    },
    {
      "feature_id": 1,
      "feature_key": "feature_000001",
      "present": false,
      "value": null
    }
  ]
}
```

Java 예제는 `present=true`이고 `value`가 null이 아닌 항목만 raw sample stage에 씁니다. 누락된 값은 dense-long shard 생성 시 `mask=0`으로 채워집니다.

## Tests

```powershell
java -cp "java\lib\*;java\out" scripts.RunScalarBuilderTestsMain
java -cp "java\lib\*;java\out" scripts.RunScalarNotebookBuilderTestsMain
```

## Reference

- format 문서: [docs/scalar_parquet_shard_format.md](../../docs/scalar_parquet_shard_format.md)
- Java 전체 설명: [java/README.md](../../java/README.md)
