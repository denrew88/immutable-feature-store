Java용 scalar feature shard 라이브러리 패키지다.

이 패키지는 dense-id scalar shard reader facade, bundled sample-major stage 기반 direct-ingestion builder, selection facade, metadata helper를 담는다.

진입점:
- `fs.io.ScalarFeatureShards`

## 준비물

- Java 8
- DuckDB JDBC jar
  - `java/lib/duckdb_jdbc-1.1.3.jar`

jar가 없으면 먼저 내려받는다.

```powershell
powershell -ExecutionPolicy Bypass -File java\download_duckdb_jdbc.ps1
```

## 빌드

```powershell
cd packages\scalar_feature_shard_java
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

출력:
- `dist/scalar-feature-shard-java-0.1.0.jar`

thin jar이므로 실행 시 DuckDB JDBC jar를 classpath에 같이 넣어야 한다.

## Public API

### `fs.io.ScalarFeatureShards`

정적 facade다. manifest 로드, metadata 작성, builder 생성, selection entrypoint를 묶는다.

- `loadManifest(...)`
  - `ShardManifest`를 읽는다.
- `open(...)`
  - `ScalarShardDataset`을 연다.
- `writeSampleMeta(...)`
  - sample metadata parquet를 쓴다.
- `writeFeatureMeta(...)`
  - feature metadata parquet를 쓴다.
- `newBuilder(...)`
  - direct-ingestion builder를 만든다.
- `buildCandidates(...)`
  - selection 후보 목록을 만든다.
- `selectFeatures(...)`
  - 최종 선택 결과를 만든다.

### `fs.io.ScalarShardDataset`

scalar shard를 읽는 reader facade다.

- `manifest()`
  - 현재 dataset의 manifest를 돌려준다.
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
  - feature 여러 개를 batch로 읽고 `ScalarFeatureValues`를 하나씩 돌려준다.
- `iterManyByKey(featureKeys, sampleKeys[, batchSize, maintainOrder])`
  - key 기반 batched iteration이다.

`maintainOrder=false`를 쓰면 feature를 shard locality 기준으로 재정렬해서 더 빠르게 읽을 수 있다.

### `fs.io.ScalarDatasetBuilder`

sample 단위 입력을 받아 sample-major stage를 만든 뒤 최종 shard를 빌드한다.

- `writeSample(sampleId, values)`
  - sample 하나를 한 번에 기록한다.
- `openSample(sampleId)`
  - context를 열고 `writeValue(...)`, `writeValues(...)`를 호출한다.
- `finishSampleMajor()`
  - intermediate sample-major stage를 마무리한다.
- `updateFeatureMeta(records, on, requireAll)`
  - discovered-feature mode 뒤에 feature metadata를 보강한다.
- `buildShards([keepSampleMajor])`
  - sample-major stage를 최종 shard artifact로 변환한다.

## 사용 예제

```java
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

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.newBuilder(
                "C:\\data\\scalar_shards",
                "C:\\data\\sample_meta.parquet",
                "C:\\data\\feature_meta.parquet")) {
            builder.writeSample(0L, row("feature_a", 1.23, "feature_b", 4.56));
            builder.writeSample(1L, row("feature_a", 7.89));
            builder.buildShards(false);
        }

        try (ScalarShardDataset ds = ScalarFeatureShards.open("C:\\data\\scalar_shards\\shard_manifest.json")) {
            ScalarFeatureValues values = ds.getValues(0, new long[]{0L, 1L});
            System.out.println(values.values().size());
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

- 최종 scalar artifact는 standalone 구조다.
- `selection_stats/<y>.parquet`를 통해 selection fast path를 지원한다.
- intermediate sample-major stage는 file-per-sample이 아니라 bundle 기반이다.
- 포맷 설명은 `docs/scalar_parquet_shard_format.md`를 본다.
