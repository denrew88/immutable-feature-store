# Java 구현 안내

이 디렉터리는 Java 8 기준의 array/scalar shard 구현, CLI, 테스트, jar 패키지 빌드 스크립트를 담고 있다.  
Maven이나 Gradle을 쓰지 않고 `javac`와 `jar`를 직접 호출하는 구조다.

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
      math/
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
  README.md
```

구조 원칙은 이렇다.

- `fs.io`
  - 외부에서 바로 쓰는 public facade와 dataset entrypoint가 있다.
- `fs.io.array`
  - array reader, bundle writer, shard builder, manifest helper가 있다.
- `fs.io.scalar`
  - scalar reader, sample-major writer, shard builder, selection용 helper가 있다.
- `fs.io.common`
  - DuckDB, JSON, dtype/배열 helper가 있다.
- `fs.model.common`
  - array/scalar가 같이 쓰는 공용 metadata/type 모델이 있다.
- `fs.model.array`
  - array binary shard 전용 모델이 있다.
- `fs.model.scalar`
  - scalar shard 전용 모델이 있다.

## 컴파일

보통 저장소 루트에서 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_java_libs.ps1
powershell -Command "Get-ChildItem java\src -Recurse -Filter *.java | ForEach-Object { $_.FullName } | Set-Content java\sources.txt"
javac -encoding UTF-8 -cp "java\lib\*" -d java\out @java\sources.txt
```

한글 Javadoc이 있으므로 `-encoding UTF-8`을 항상 넣는 것이 중요하다.

## Java 패키지

### Array jar

패키지 위치:
- `packages/array_binary_shard_java`

빌드:

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_binary_shard_java\build.ps1
```

산출물:
- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0.jar`
- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0-sources.jar`
- `packages/array_binary_shard_java/dist/array-binary-shard-java-0.3.0-javadoc.jar`

### Scalar jar

패키지 위치:
- `packages/scalar_feature_shard_java`

빌드:

```powershell
powershell -ExecutionPolicy Bypass -File packages\scalar_feature_shard_java\build.ps1
```

산출물:
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-sources.jar`
- `packages/scalar_feature_shard_java/dist/scalar-feature-shard-java-0.1.0-javadoc.jar`

두 jar 모두 thin jar이므로 실행할 때 DuckDB JDBC와 Jackson jar를 classpath에 같이 넣어야 한다.

## Array CLI

기존 bundle manifest에서 array binary shard를 만든다.

```powershell
java -cp "java\lib\*;java\out" scripts.BuildArrayShardsMain ^
  --bundle-manifest C:\data\array_bundle_manifest.json ^
  --out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16
```

synthetic array 데이터를 생성하고 바로 shard까지 만든다.

```powershell
java -cp "java\lib\*;java\out" scripts.GenerateArraySynthMain ^
  --bundle-out-dir C:\data\array_bundles ^
  --sample-meta C:\data\array_sample_meta.parquet ^
  --shard-out-dir C:\data\array_shards ^
  --target-shard-mb 32 ^
  --samples-per-block 16 ^
  --n-samples 256 ^
  --n-features 64 ^
  --seed 7
```

array feature가 어느 shard와 block에 들어 있는지 본다.

```powershell
java -cp "java\lib\*;java\out" scripts.LocateArrayFeatureMain ^
  --manifest C:\data\array_shards\array_binary_shard_manifest.json ^
  --feature-id 12345
```

## Array Public API

### 진입점

- `fs.io.ArrayBinaryShards`
- `fs.io.ArrayBinaryShardReader`
- `fs.io.ArrayDatasetBuilder`

### `fs.io.ArrayBinaryShards`

정적 facade다. metadata I/O, builder 생성, reader 진입점을 모아둔다.

- `buildFromBundles(bundleManifestPath, outDir, config)`
  - 기존 array bundle stage를 최종 binary shard artifact로 변환한다.
- `loadManifest(manifestPath)`
  - `ArrayShardManifest`를 읽는다.
- `open(manifestPath)`
  - `ArrayBinaryShardReader`를 연다.
- `loadLocator(manifest)`
  - `feature_id -> shard/block 위치` 인덱스를 연다.
- `loadSampleIds(manifest)`
  - `sample_id <-> sample_key` 인덱스를 연다.
- `loadFeatureIds(manifest)`
  - `feature_id <-> feature_key` 인덱스를 연다.
- `writeSampleMeta(records, path)`
  - `List<Map<String, Object>>` 형태의 sample metadata를 parquet로 쓴다.
- `writeFeatureMeta(records, path)`
  - `List<Map<String, Object>>` 형태의 feature metadata를 parquet로 쓴다.
- `newBuilder(...)`
  - direct-ingestion builder를 만든다.

### `fs.io.ArrayBinaryShardReader`

binary shard를 실제로 읽는 저수준 reader다.

- `pointSchema()`
  - manifest의 point schema를 돌려준다.
- `loadBlock(shardId, rowInShard[, decodeCategorical])`
  - shard 안의 block payload 하나를 로드한다.
- `loadFeatureSamples(featureId, sampleIds, locatorIndex[, decodeCategorical])`
  - feature 하나에 대해 지정한 sample들의 trace를 읽는다.
- `loadFeatureSamplesBySampleIds(featureId, sampleIds, sampleIdIndex, locatorIndex[, decodeCategorical])`
  - sample id 기반 조회를 helper index와 함께 수행한다.
- `loadFeatureSamplesBySampleKeys(featureId, sampleKeys, sampleIdIndex, locatorIndex[, decodeCategorical])`
  - sample key 기반 조회를 수행한다.
- `loadFeatureSamplesByKeys(featureKey, sampleKeys, featureIdIndex, sampleIdIndex, locatorIndex[, decodeCategorical])`
  - feature key와 sample key 모두 외부 키로 조회한다.

### `fs.io.ArrayDatasetBuilder`

bundle을 직접 만들지 않고 trace를 바로 넣어 최종 shard까지 만드는 builder다.

- 생성자
  - known-feature mode와 discovered-feature mode를 모두 지원한다.
- `addTrace(sampleId, featureId, featureKey, columns)`
  - sample 하나에 trace 하나를 추가한다.
- `sample(sampleId)`
  - sample context를 열고 그 안에서 `addTrace(...)`를 호출한다.
- `finishBundles()`
  - intermediate bundle stage를 명시적으로 마무리한다.
- `updateFeatureMeta(records, on, requireAll)`
  - discovered-feature mode 뒤에 extra feature metadata를 덧붙인다.
- `buildShards([cleanupBundles])`
  - bundle stage를 최종 array binary shard로 변환한다.

### Array builder 예제

```java
import fs.io.ArrayBinaryShards;
import fs.io.ArrayDatasetBuilder;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public class ArrayExample {
    public static void main(String[] args) throws Exception {
        ArrayBinaryShards.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000", "split", "train"),
                        row("sample_key", "sample_000001", "split", "test")
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
                    row("phase", new int[]{10, 11, 12}, "state_code", new String[]{"OK", "OK", "WARN"})
            );

            builder.buildShards(true);
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

## Array Binary v3 요약

자바 array 구현은 Python array binary shard v3와 같은 기준을 따른다.

핵심:

- standalone artifact 구조
  - `array_binary_shard_manifest.json`
  - `sample_meta.parquet`
  - `feature_meta.parquet`
  - 필요 시 `categorical_dictionaries/*.json`
  - `array_binary_feature_shards/shard_XXXX.blocks.idx`
  - `array_binary_feature_shards/shard_XXXX.blocks.bin`
- dense id
  - `sample_id == sample_meta.parquet` row index
  - `feature_id == feature_meta.parquet` row index
- `point_schema`는 manifest가 정의
- `time`, `value`는 필수가 아님
- categorical은 integer code로 저장하고 reader가 label로 decode 가능

## Scalar CLI

`sample_meta.parquet`와 `feature_meta.parquet`에서 바로 scalar shard를 만든다.

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-meta C:\data\sample_meta.parquet ^
  --feature-meta C:\data\feature_meta.parquet ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

bundle 기반 sample-major stage에서 scalar shard를 만든다.

```powershell
java -cp "java\lib\*;java\out" scripts.BuildShardsMain ^
  --sample-bundle-manifest C:\data\sample_major_stage\sample_major_manifest.json ^
  --out-dir C:\data\shards ^
  --target-shard-mb 32
```

selection을 실행한다.

```powershell
java -cp "java\lib\*;java\out" scripts.RunSelectionMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --top-m 100
```

scalar feature 하나의 위치를 확인한다.

```powershell
java -cp "java\lib\*;java\out" scripts.LocateFeatureMain ^
  --manifest C:\data\shards\shard_manifest.json ^
  --feature-id 12345
```

## Scalar Public API

### 진입점

- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarShardDataset`
- `fs.io.ScalarDatasetBuilder`

### `fs.io.ScalarFeatureShards`

정적 facade다. manifest 로드, metadata 작성, builder 생성, selection entrypoint를 묶는다.

- `loadManifest(manifestPath)`
  - `ShardManifest`를 읽는다.
- `open(manifestPath)`
  - `ScalarShardDataset`을 연다.
- `writeSampleMeta(records, path)`
  - sample metadata를 parquet로 쓴다.
- `writeFeatureMeta(records, path)`
  - feature metadata를 parquet로 쓴다.
- `newBuilder(outDir, sampleMetaPath[, featureMetaPath])`
  - scalar direct-ingestion builder를 만든다.
- `buildCandidates(manifestPath, yCol, config)`
  - selection 후보 점수 목록을 만든다.
- `selectFeatures(manifestPath, yCol, config)`
  - 최종 선택된 후보 목록을 만든다.

### `fs.io.ScalarShardDataset`

scalar shard를 여는 reader facade다.

- `manifest()`
  - 열려 있는 dataset의 manifest를 돌려준다.
- `getValue(featureId, sampleId)`
  - 값 하나를 읽는다.
- `getValueByKey(featureKey, sampleKey)`
  - 외부 키 기준으로 값 하나를 읽는다.
- `getValues(featureId, sampleIds)`
  - feature 하나에 대해 sample 여러 개를 읽는다.
- `getValuesBySampleKeys(featureId, sampleKeys)`
  - feature id + sample key 기준 조회다.
- `getValuesByKeys(featureKey, sampleKeys)`
  - feature key + sample key 기준 조회다.
- `iterMany(featureIds, sampleIds[, batchSize, maintainOrder])`
  - feature 여러 개를 배치 단위로 읽고 `ScalarFeatureValues`를 순서대로 돌려준다.
- `iterManyByKey(featureKeys, sampleKeys[, batchSize, maintainOrder])`
  - key 기반 batched iteration이다.

`maintainOrder=false`를 쓰면 feature를 shard locality 기준으로 재정렬해서 더 빠르게 읽을 수 있다.

### `fs.io.ScalarDatasetBuilder`

sample 단위 입력을 받아 intermediate sample-major stage를 만들고 최종 shard를 빌드한다.

- `writeSample(sampleId, values)`
  - sample 하나의 feature map을 한 번에 쓴다.
- `openSample(sampleId)`
  - `ScalarSampleContext`를 열고 `writeValue(...)`, `writeValues(...)`를 호출한다.
- `finishSampleMajor()`
  - intermediate sample-major stage를 명시적으로 마무리한다.
- `updateFeatureMeta(records, on, requireAll)`
  - discovered-feature mode 뒤에 extra feature metadata를 덧붙인다.
- `buildShards([keepSampleMajor])`
  - sample-major stage를 최종 scalar shard artifact로 변환한다.

### Scalar builder 예제

```java
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarFeatureShards;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ScalarExample {
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

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.newBuilder(
                "C:\\data\\scalar_shards",
                "C:\\data\\sample_meta.parquet",
                "C:\\data\\feature_meta.parquet")) {
            builder.writeSample(0L, row("feature_a", 1.23, "feature_b", 4.56));
            builder.writeSample(1L, row("feature_a", 7.89));
            builder.buildShards(false);
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

- `java/lib/*.jar`는 git 추적 대상이 아니다.
- array binary codec 기본값은 `none`이다.
- array binary shard의 기본 block 크기는 `samples_per_block=16`이다.
- shard 분할 기본값은 `target_shard_mb=32`다.
- `blocks.idx`는 `(feature_id, block_id)` 조합마다 32바이트 고정 record를 가진다.
- `blocks.bin`은 가변 길이 block payload를 저장한다.
