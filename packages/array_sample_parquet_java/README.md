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

Java builder는 Python 표준 builder와 같은 2단계 구조입니다. sample이 닫히면 먼저 `raw_samples/sample_*.parquet`와 `raw_trace_index/sample_*.parquet`를 확정하고, `raw_samples.jsonl`에 commit record를 append합니다. `finish()` 또는 `compact()`는 raw 파일들을 `targetPartBytes`, `maxPartRows`, `maxPartSamples` 기준으로 묶어서 최종 `sample_parts`와 `trace_index_parts`를 만듭니다.

raw sample write는 Java 8 호환 Apache Arrow vector batch를 DuckDB에 `registerArrowStream(...)`으로 넘기고, sample close 시 `COPY ... TO parquet`로 raw 파일을 씁니다. 이 경로는 point row마다 `DuckDBAppender`를 호출하지 않습니다.

사용자가 `addTrace(...)`를 feature 순서대로 호출한다는 보장은 없으므로, sample close 직전에 Java가 trace 목록을 `(sample_id, feature_id)` 순서로 정렬합니다. 그 뒤 raw write와 compact는 DuckDB SQL `ORDER BY` 없이 이미 정렬된 stream/file 목록을 그대로 `COPY TO parquet`로 씁니다. `ArraySampleParquetOrderChecks`는 raw/final parquet의 물리 row 순서가 실제로 `(sample_id, feature_id, point_idx)` 또는 `(sample_id, feature_id)`인지 검사합니다.

중간에 종료되면 `raw_samples.jsonl`에 기록된 sample만 완료로 인정합니다. resume 시 `.tmp` 파일을 삭제하고 `status().pendingSampleIds`를 worker에게 나눠주면 됩니다. 순차 실행을 원하면 이 목록을 앞에서부터 처리하면 됩니다.

`ArraySampleParquetBuildOptions.arrowBatchRows`는 DuckDB로 넘기는 Arrow record batch의 최대 point row 수입니다. 기본값은 `262144`이며, 값을 키우면 batch 수가 줄지만 sample close 시점의 off-heap buffer 사용량이 커집니다.

### File lock behavior

raw stage는 sample parquet, trace index, commit log 경계를 `.lock` 파일로 보호합니다. lock 파일에는 `token`, `pid`, `thread`, `host`, `created_at_ms`가 기록됩니다. release 시에는 token이 같은 경우에만 삭제하며, Windows 백신/인덱서가 잠깐 파일을 잡는 경우를 고려해 짧은 backoff로 삭제를 재시도합니다. 최종 release 실패는 조용히 무시하지 않고 예외로 드러납니다.

프로세스가 죽어 stale lock이 남은 경우 기본값은 자동 삭제하지 않는 것입니다. 필요할 때만 JVM system property `-Dfs.fileLockStaleMillis=<millis>` 또는 환경변수 `FS_FILE_LOCK_STALE_MILLIS`로 켤 수 있습니다. stale 삭제는 같은 host이고, lock 파일의 pid가 살아있지 않으며, 삭제 직전 token이 그대로인 경우에만 시도합니다.

## Config Guide

처음에는 아래 설정만 넣으면 됩니다.

```java
ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
options.targetPartBytes = 128L * 1024L * 1024L;
options.compression = "zstd";
```

| option | 기본값 | 설명 |
| --- | --- | --- |
| `targetPartBytes` | 128MB | final sample part 하나의 목표 크기입니다. part가 너무 많으면 키우고, 한 part가 너무 크면 줄입니다. |
| `maxPartRows` | 10,000,000 | point row 수 기준 안전장치입니다. trace가 매우 길 때 줄일 수 있습니다. |
| `maxPartSamples` | 0 | part 하나의 최대 sample 수입니다. 0이면 sample 수로 제한하지 않습니다. |
| `compression` | `"zstd"` | parquet compression입니다. 디버깅/속도 확인은 `"none"`, 저장 용량은 `"zstd"`를 씁니다. |
| `sampleKeyCol` | `"sample_key"` | sample metadata의 key column 이름이 다를 때만 바꿉니다. |
| `featureKeyCol` | `"feature_key"` | feature metadata의 key column 이름이 다를 때만 바꿉니다. |
| `duckdbThreads` | 0 | DuckDB writer thread 수입니다. 0이면 DuckDB 기본값입니다. |
| `arrowBatchRows` | 262,144 | Java raw sample writer가 DuckDB로 넘기는 Arrow batch의 최대 point row 수입니다. 크면 batch overhead는 줄지만 off-heap memory 사용량이 커집니다. |

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packages\array_sample_parquet_java\build.ps1
```

산출물:

- `dist/array-sample-parquet-java-0.1.0.jar`
- `dist/array-sample-parquet-java-0.1.0-sources.jar`
- `dist/array-sample-parquet-java-0.1.0-javadoc.jar`

thin jar이므로 실행 시 필요한 runtime jar를 classpath에 같이 넣어야 합니다. 현재 구현의 최소 runtime dependency는 다음과 같습니다.

- `duckdb_jdbc-1.1.3.jar`
- `jackson-core-2.20.0.jar`
- `jackson-databind-2.20.0.jar`
- `jackson-annotations-2.20.jar`
- `arrow-c-data-14.0.2.jar`
- `arrow-memory-core-14.0.2.jar`
- `arrow-memory-unsafe-14.0.2.jar`
- `arrow-vector-14.0.2-shade-format-flatbuffers.jar`
- `netty-common-4.1.96.Final.jar`
- `slf4j-api-1.7.36.jar`

Hadoop/Parquet Java writer jar는 필요하지 않습니다. parquet 파일 생성과 `zstd` 압축은 DuckDB JDBC가 수행합니다.

Arrow 관련 jar는 raw sample write fast path에 필요합니다. Java trace 배열을 Arrow vector batch로 묶고 `ArrowArrayStream`을 DuckDB `registerArrowStream(...)`에 등록한 뒤, DuckDB `COPY ... TO parquet`로 파일을 씁니다. 이 구조 덕분에 point row마다 JDBC append를 호출하지 않습니다. `slf4j-api`는 Arrow의 logging API 의존성입니다. 별도 binding이 없으면 SLF4J NOP 경고가 출력될 수 있지만 실행에는 문제가 없습니다.

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
    for (Long sampleId : builder.status().pendingSampleIds) {
        try (ArraySampleParquetSampleContext sample = builder.sample(sampleId.longValue(), true)) {
            if (sample.skipped) {
                continue;
            }
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
- `examples/BuildArraySampleParquetFromValueApiWithJarExample.java`: 이미 존재하는 sample/feature meta를 기준으로 Python value API를 호출해서 전체 part를 생성합니다.

컴파일:

```powershell
New-Item -ItemType Directory -Force packages\array_sample_parquet_java\examples\out | Out-Null
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  -d packages\array_sample_parquet_java\examples\out `
  packages\array_sample_parquet_java\examples\BuildArraySampleParquetWithJarExample.java
```

Python value API 호출 예제를 컴파일하려면 마지막 파일명만 바꾸면 됩니다.

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\javac.exe" `
  -encoding UTF-8 `
  -cp "packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  -d packages\array_sample_parquet_java\examples\out `
  packages\array_sample_parquet_java\examples\BuildArraySampleParquetFromValueApiWithJarExample.java
```

실행:

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\array_sample_parquet_java\examples\out;packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  BuildArraySampleParquetWithJarExample
```

기본 출력 위치는 `data/tmp_array_sample_parquet_jar_example`입니다. 다른 위치에 쓰려면 실행 명령 끝에 출력 root directory를 인자로 넘기면 됩니다.

Python value API 호출 예제는 `python/scripts/serve_synthetic_value_api.py`가 먼저 떠 있어야 하며, sample meta와 feature meta는 이미 생성되어 있어야 합니다.

```powershell
& "C:\Program Files\Java\jdk-1.8\bin\java.exe" `
  -cp "packages\array_sample_parquet_java\examples\out;packages\array_sample_parquet_java\dist\array-sample-parquet-java-0.1.0.jar;java\lib\*" `
  BuildArraySampleParquetFromValueApiWithJarExample `
  --base-url http://127.0.0.1:8010 `
  --sample-meta data\sample_meta.parquet `
  --feature-meta data\feature_meta.parquet `
  --out-dir data\array_sample_parquet_from_api
```

API 서버 실행:

```powershell
python python\scripts\serve_synthetic_value_api.py --host 127.0.0.1 --port 8010
```

`BuildArraySampleParquetFromValueApiWithJarExample`는 sample 하나와 feature id 묶음마다 `POST /array/traces`를 호출합니다. 요청은 `sample_id` 또는 `sample_key` 중 하나, `feature_ids` 또는 `feature_keys` 중 하나만 허용합니다.

요청 예:

```json
{
  "sample_meta_path": "data/sample_meta.parquet",
  "feature_meta_path": "data/feature_meta.parquet",
  "sample_id": 0,
  "feature_ids": [0, 1, 2],
  "seed": 7,
  "include_missing": false,
  "min_trace_len": 24,
  "max_trace_len": 48
}
```

응답 예:

```json
{
  "sample_id": 0,
  "sample_key": "sample_000000",
  "feature_count": 3,
  "trace_count": 3,
  "point_schema": [
    {"name": "time", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "value", "storage_type": "float64", "logical_type": "continuous"},
    {"name": "ch_step", "storage_type": "string", "logical_type": "categorical"}
  ],
  "traces": [
    {
      "feature_id": 0,
      "feature_key": "feature_000000",
      "present": true,
      "trace_len": 24,
      "columns": {
        "time": [0.0, 0.4347826087],
        "value": [0.12, 0.18],
        "ch_step": ["pre", "pre"]
      }
    }
  ]
}
```

`present=false` trace는 `include_missing=true`일 때만 응답에 포함됩니다. 이 경우 `trace_len=0`이고 `columns`는 빈 배열입니다.

## When To Use

이 jar는 sample 중심 viewer/debugging과 Java 쪽 검증 스크립트에 적합합니다. feature-major serving이나 대량 random access가 목적이면 기존 custom binary array shard를 사용하십시오.

자세한 포맷, 구현 방식, API 서버 요청/응답은 `docs/array_sample_parquet_format_v1.md`를 참고하십시오.
