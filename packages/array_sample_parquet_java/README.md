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

## Format Summary

artifact 구조는 다음과 같습니다.

```text
out_dir/
  array_sample_parquet_manifest.json
  state.json
  parts.jsonl
  sample_meta.parquet
  feature_meta.parquet
  sample_parts/
    part_000000.parquet
  categorical_dictionaries/
    ch_step.json
```

`sample_parts/*.parquet`의 row 하나는 `(sample_id, feature_id)` trace 하나입니다. point column은 Parquet LIST column입니다. missing trace는 row 부재, empty trace는 `trace_len=0` row로 표현합니다.

## Build Behavior

Java builder도 전체 part를 메모리에 들고 있지 않습니다. `addTrace(...)`가 호출되면 `ArraySampleParquetPartWriter`가 parquet-hadoop `ExampleParquetWriter`로 `.parquet.tmp`에 trace row를 바로 씁니다.

part commit은 sample 경계에서만 발생합니다. 기본 flush 기준은 `targetPartBytes`이고, `maxPartRows`와 `maxPartSamples`는 안전장치입니다. commit 시 `.parquet.tmp`를 `.parquet`으로 rename하고 `parts.jsonl`, `state.json`을 갱신합니다.

중간에 종료되면 `parts.jsonl`에 기록된 part만 committed로 인정합니다. resume 시 `.tmp` 파일을 삭제하고 `status().nextExpectedSampleId`부터 다시 입력하면 됩니다.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_sample_parquet_java\build.ps1
```

산출물:

- `dist/array-sample-parquet-java-0.1.0.jar`
- `dist/array-sample-parquet-java-0.1.0-sources.jar`
- `dist/array-sample-parquet-java-0.1.0-javadoc.jar`

thin jar이므로 실행 시 `java/lib/*.jar`를 classpath에 같이 넣어야 합니다.

## Example

```java
ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
options.targetPartBytes = 128L * 1024L * 1024L;
options.maxPartRows = 100000;
options.maxPartSamples = 0;
options.compression = "zstd";

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

## Jar Example

sample meta, feature meta, array sample parquet dataset을 jar classpath만으로 생성하는 전체 예제는 다음 파일에 있습니다.

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
