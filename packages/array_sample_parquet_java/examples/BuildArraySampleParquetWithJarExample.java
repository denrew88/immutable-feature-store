import fs.io.ArraySampleParquets;
import fs.io.array_sample_parquet.ArraySampleParquetDatasetBuilder;
import fs.io.array_sample_parquet.ArraySampleParquetReader;
import fs.io.array_sample_parquet.ArraySampleParquetSampleContext;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array-sample-parquet-java jar만 classpath에 넣어서 sample meta, feature meta,
 * array_sample_parquet dataset을 생성하는 최소 예제다.
 *
 * <p>실행 결과는 기본적으로 data/tmp_array_sample_parquet_jar_example 아래에 생성된다.
 * 첫 번째 인자로 출력 root directory를 넘기면 그 경로를 사용한다.
 */
public class BuildArraySampleParquetWithJarExample {
    public static void main(String[] args) throws Exception {
        File root = new File(args.length > 0 ? args[0] : "data/tmp_array_sample_parquet_jar_example").getAbsoluteFile();
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create root dir: " + root.getAbsolutePath());
        }

        // 1. sample metadata를 만든다. row 순서가 dense sample_id가 된다.
        String sampleMetaPath = ArraySampleParquets.writeSampleMeta(
                Arrays.asList(
                        row("sample_key", "sample_000000", "split", "train"),
                        row("sample_key", "sample_000001", "split", "train"),
                        row("sample_key", "sample_000002", "split", "valid")
                ),
                new File(root, "sample_meta.parquet").getAbsolutePath()
        );

        // 2. feature metadata를 만든다. row 순서가 dense feature_id가 된다.
        String featureMetaPath = ArraySampleParquets.writeFeatureMeta(
                Arrays.asList(
                        row("feature_key", "feature_a", "group", "alpha"),
                        row("feature_key", "feature_b", "group", "beta")
                ),
                new File(root, "feature_meta.parquet").getAbsolutePath()
        );

        // 3. trace point schema를 정의한다.
        List<PointColumnSpec> pointSchema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL)
        );

        ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
        options.targetPartBytes = 1024L * 1024L;
        options.maxPartRows = 10000000;
        options.maxPartSamples = 0;
        options.compression = "none";

        File datasetDir = new File(root, "array_sample_parquet");
        String manifestPath;

        // 4. sample 순서대로 trace를 넣고 dataset을 완성한다.
        try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
                datasetDir.getAbsolutePath(),
                sampleMetaPath,
                pointSchema,
                featureMetaPath,
                options)) {
            long start = builder.status().nextExpectedSampleId;
            for (long sampleId = start; sampleId < 3L; sampleId++) {
                try (ArraySampleParquetSampleContext sample = builder.sample(sampleId)) {
                    if (sampleId == 0L) {
                        sample.addTrace(null, "feature_a", columns(
                                new double[]{0.0, 1.0, 2.0},
                                new double[]{10.0, 11.0, 12.0},
                                new String[]{"idle", "run", "run"}));
                        sample.addTrace(null, "feature_b", columns(
                                new double[]{0.0, 1.0},
                                new double[]{20.0, 21.0},
                                new String[]{"idle", "idle"}));
                    } else if (sampleId == 1L) {
                        sample.addTrace(null, "feature_a", columns(
                                new double[]{0.0},
                                new double[]{30.0},
                                new String[]{"stop"}));
                    } else {
                        // sample_000002는 feature_b trace만 가진다. feature_a는 missing trace가 된다.
                        sample.addTrace(null, "feature_b", columns(
                                new double[]{},
                                new double[]{},
                                new String[]{}));
                    }
                }
            }
            manifestPath = builder.finish();
        }

        // 5. jar reader로 간단히 검증한다.
        try (ArraySampleParquetReader reader = ArraySampleParquets.open(manifestPath)) {
            List<ArraySampleParquetTrace> traces = reader.loadTracesByKeys(
                    new String[]{"sample_000000", "sample_000001", "sample_000002"},
                    new String[]{"feature_a", "feature_b"},
                    true,
                    true);
            System.out.println("trace_count=" + traces.size());
            for (ArraySampleParquetTrace trace : traces) {
                System.out.println(
                        "sample_id=" + trace.sampleId
                                + ", feature_id=" + trace.featureId
                                + ", present=" + trace.present
                                + ", trace_len=" + trace.traceLen
                                + ", columns=" + trace.columns.keySet());
            }
        }

        System.out.println("sample_meta=" + sampleMetaPath);
        System.out.println("feature_meta=" + featureMetaPath);
        System.out.println("manifest=" + manifestPath);
    }

    private static LinkedHashMap<String, Object> columns(double[] time, double[] value, String[] chStep) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        out.put("time", time);
        out.put("value", value);
        out.put("ch_step", chStep);
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            row.put(kvs[i].toString(), kvs[i + 1]);
        }
        return row;
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }
}
