# array-sample-parquet-java

`array-sample-parquet-java`는 Java 8에서 `array_sample_parquet` dataset을 생성하고 읽기 위한 thin jar 패키지입니다. 포맷은 Python 구현과 동일하며, sample-major Parquet part와 JSON manifest/state를 사용합니다.

## Public API

- `fs.io.ArraySampleParquets`
  - `writeSampleMeta(...)`
  - `writeFeatureMeta(...)`
  - `openSession(...)`
  - `loadManifest(...)`
  - `open(...)`
- `fs.io.array_sample_parquet.ArraySampleParquetDatasetBuilder`
  - `status()`
  - `sample(sampleId)`
  - `sample(sampleKey)`
  - `finish()`
- `fs.io.array_sample_parquet.ArraySampleParquetSampleContext`
  - `addTrace(featureId, featureKey, columns)`
- `fs.io.array_sample_parquet.ArraySampleParquetReader`
  - `loadTracesByIds(...)`
  - `loadTracesByKeys(...)`
- `fs.io.array_sample_parquet.ArraySampleParquetOrderChecks`
  - `requirePointRowsSorted(...)`
  - `requireTraceIndexRowsSorted(...)`

## Format Summary
Long-format physical layout:

- `sample_parts/*.parquet`: point rows with `sample_id`, `feature_id`, `point_idx`, and primitive point columns.
- `trace_index_parts/*.parquet`: present trace rows with `sample_id`, `feature_id`, `trace_len`.
- Empty trace is `trace_len=0` in trace index with no point row; missing trace has no trace index row.

artifact 구조는 다음과 같습니다.

```text
out_dir/
  array_sample_parquet_manifest.json
  raw_state.json
  raw_samples.jsonl
  sample_meta.parquet
  feature_meta.parquet
  raw_samples/
    sample_000000000000.parquet
  raw_trace_index/
    sample_000000000000.parquet
  sample_parts/
    part_000000.parquet
  trace_index_parts/
    part_000000.parquet
```

`sample_parts/*.parquet`의 row 하나는 trace point 하나입니다. 물리 schema는 `sample_id`, `feature_id`, `point_idx`, point columns로 구성됩니다. `trace_index_parts/*.parquet`에는 present trace마다 `(sample_id, feature_id, trace_len)`이 기록됩니다. missing trace는 trace index row 부재, empty trace는 `trace_len=0`인 trace index row와 point row 부재로 표현합니다.

categorical point column은 별도 dictionary sidecar 없이 string primitive로 저장합니다. 반복 문자열 압축은 Parquet의 dictionary/RLE encoding에 맡깁니다.

## Build Behavior

Java builder는 Python raw builder와 같은 2단계 구조입니다. sample이 닫히면 먼저 `raw_samples/sample_*.parquet`와 `raw_trace_index/sample_*.parquet`를 확정하고, `raw_samples.jsonl`에 commit record를 append합니다. `finish()` 또는 `compact()`는 raw 파일들을 `targetPartBytes`, `maxPartRows`, `maxPartSamples` 기준으로 묶어서 최종 `sample_parts`와 `trace_index_parts`를 만듭니다.

raw sample write는 Java 8 호환 Apache Arrow vector batch를 DuckDB에 `registerArrowStream(...)`으로 넘기고, sample close 시 `COPY ... TO parquet`로 raw 파일을 씁니다. 이 경로는 point row마다 `DuckDBAppender`를 호출하지 않습니다.

사용자가 `addTrace(...)`를 feature 순서대로 호출한다는 보장은 없으므로, sample close 직전에 Java가 trace 목록을 `(sample_id, feature_id)` 순서로 정렬합니다. 그 뒤 raw write와 compact는 DuckDB SQL `ORDER BY` 없이 이미 정렬된 stream/file 목록을 그대로 `COPY TO parquet`로 씁니다. `ArraySampleParquetOrderChecks`는 raw/final parquet의 물리 row 순서가 실제로 `(sample_id, feature_id, point_idx)` 또는 `(sample_id, feature_id)`인지 검사합니다.

중간에 종료되면 `raw_samples.jsonl`에 기록된 sample만 완료로 인정합니다. resume 시 `.tmp` 파일을 삭제하고 `status().pendingSampleIds`를 worker에게 나눠주면 됩니다. 기존 순차 예제와 호환되도록 `status().nextExpectedSampleId`는 가장 작은 pending sample id를 반환합니다.

`ArraySampleParquetBuildOptions.arrowBatchRows`는 DuckDB로 넘기는 Arrow record batch의 최대 point row 수입니다. 기본값은 `262144`이며, 값을 키우면 batch 수가 줄지만 sample close 시점의 off-heap buffer 사용량이 커집니다.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_sample_parquet_java\build.ps1
```

산출물:

- `dist/array-sample-parquet-java-0.1.0.jar`
- `dist/array-sample-parquet-java-0.1.0-sources.jar`
- `dist/array-sample-parquet-java-0.1.0-javadoc.jar`

thin jar이므로 실행 시 `java/lib/*.jar`를 classpath에 같이 넣어야 합니다.

현재 표준 경로에 필요한 추가 런타임 jar는 DuckDB/Jackson/Hadoop/Parquet 기본 jar 외에 다음 Arrow vector bridge jar입니다.

- `arrow-c-data-14.0.2.jar`
- `arrow-memory-core-14.0.2.jar`
- `arrow-memory-unsafe-14.0.2.jar`
- `arrow-vector-14.0.2-shade-format-flatbuffers.jar`
- `netty-common-4.1.96.Final.jar`

## Performance Notes

로컬 기준 `20 samples x 1200 features x trace_len 950`, point columns `time/value float64 + ch_step string`, `zstd` 조건에서 최근 측정 결과는 다음과 같습니다.

- 전체 build: 약 `8.2s ~ 10.0s`
- raw sample close/write 합계: 약 `4.9s ~ 6.2s`
- compact finish: 약 `2.9s ~ 3.3s`
- raw samples: 약 `166.47MB`
- final sample parts: 약 `166.38MB`

같은 산출물에 대해 raw sample, raw trace index, final sample parts, final trace index 모두 물리 row 정렬 검사를 통과했습니다.

## Example

```java
ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
options.targetPartBytes = 128L * 1024L * 1024L;
options.maxPartRows = 10000000;
options.maxPartSamples = 0;
options.compression = "zstd";
options.arrowBatchRows = 262144;

try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
        "data/array_sample_parquet",
        "data/sample_meta.parquet",
        pointSchema,
        "data/feature_meta.parquet",
        options)) {
    long start = builder.status().nextExpectedSampleId;
    for (long sampleId = start; sampleId < nSamples; sampleId++) {
        try (ArraySampleParquetSampleContext sample = builder.sample(sampleId)) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
    String manifestPath = builder.finish();
}
```

out-of-order 또는 worker 분배형 사용에서는 `pendingSampleIds`를 직접 쓰는 편이 더 명확합니다.

```java
for (Long sampleId : builder.status().pendingSampleIds) {
    try (ArraySampleParquetSampleContext sample = builder.sample(sampleId.longValue(), true)) {
        if (!sample.skipped) {
            sample.addTrace(null, "feature_a", columns);
        }
    }
}
String manifestPath = builder.compact();
```

## Jar Example

sample meta, feature meta, raw sample stage 재개, 최종 array sample parquet dataset 생성을 jar classpath만으로 실행하는 전체 예제는 다음 파일에 있습니다.

- `examples/BuildArraySampleParquetWithJarExample.java`

컴파일:

```powershell
New-Item -ItemType Directory -Force packages\array_sample_parquet_java\examples\out | Out-Null
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  -d packages\array_sample_parquet_java\examples\out `
  packages\array_sample_parquet_java\examples\BuildArraySampleParquetWithJarExample.java
```

실행:

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\array_sample_parquet_java\examples\out;packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  BuildArraySampleParquetWithJarExample
```

기본 출력 위치는 `data/tmp_array_sample_parquet_jar_example`입니다. 다른 위치에 쓰려면 실행 명령 끝에 출력 root directory를 인자로 넘기면 됩니다.

## When To Use

이 jar는 sample 중심 viewer/debugging과 Java 쪽 검증 스크립트에 적합합니다. feature-major serving이나 대량 random access가 목적이면 기존 custom binary array shard를 사용하십시오.

자세한 포맷, 구현 방식, API 서버 요청/응답은 `docs/array_sample_parquet_format_v1.md`를 참고하십시오.
