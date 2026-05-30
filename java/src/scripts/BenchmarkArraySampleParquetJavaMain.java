package scripts;

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
 * array_sample_parquet Java 성능 측정 스크립트.
 *
 * <p>Python benchmark와 같은 크기의 데이터를 사용한다. Java builder도 Python raw builder처럼
 * sample close 시점에 raw sample parquet를 만들고, finish 시점에 compact한다.</p>
 */
public class BenchmarkArraySampleParquetJavaMain {
    private static final int N_SAMPLES = 20;
    private static final int N_FEATURES = 1200;
    private static final int TRACE_LEN = 950;
    private static final int MAX_PART_ROWS = 10000000;

    public static void main(String[] args) throws Exception {
        if (args.length >= 2 && "--query-only".equals(args[0])) {
            runQueries(args[1]);
            return;
        }
        String compression = args.length > 0 ? args[0] : "zstd";
        File root = new File("data/tmp_java_array_sample_parquet_bench_20x1200x950_" + compression);
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create benchmark dir: " + root.getAbsolutePath());
        }

        long metaStart = System.nanoTime();
        String sampleMetaPath = ArraySampleParquets.writeSampleMeta(sampleRows(), new File(root, "sample_meta.parquet").getAbsolutePath());
        String featureMetaPath = ArraySampleParquets.writeFeatureMeta(featureRows(), new File(root, "feature_meta.parquet").getAbsolutePath());
        double metaSec = secondsSince(metaStart);

        List<PointColumnSpec> schema = Arrays.asList(
                new PointColumnSpec("time", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL)
        );
        ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
        options.targetPartBytes = 128L * 1024L * 1024L;
        options.maxPartRows = MAX_PART_ROWS;
        options.maxPartSamples = 0;
        options.compression = compression;

        double[] time = linspace(0.0d, 1.0d, TRACE_LEN);
        double[] sampleOffsets = new double[N_SAMPLES];
        double[] featureBias = new double[N_FEATURES];
        double[][] featureWaves = new double[N_FEATURES][TRACE_LEN];
        for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
            sampleOffsets[sampleId] = ((sampleId % 97) - 48.0d) * 0.001d;
        }
        for (int featureId = 0; featureId < N_FEATURES; featureId++) {
            featureBias[featureId] = ((featureId % 131) - 65.0d) * 0.0005d;
            double freq = 1.0d + (featureId % 37) * 0.031d;
            double phase = (featureId % 19) * 0.17d;
            for (int pointIdx = 0; pointIdx < TRACE_LEN; pointIdx++) {
                featureWaves[featureId][pointIdx] = Math.sin(freq * time[pointIdx] + phase);
            }
        }
        String[] chStep = new String[TRACE_LEN];
        for (int i = 0; i < TRACE_LEN; i++) {
            switch (i % 4) {
                case 0:
                    chStep[i] = "idle";
                    break;
                case 1:
                    chStep[i] = "ramp";
                    break;
                case 2:
                    chStep[i] = "hold";
                    break;
                default:
                    chStep[i] = "cool";
                    break;
            }
        }

        File outDir = new File(root, "dataset");
        long buildStart = System.nanoTime();
        double valueGenerationSec = 0.0d;
        double addTraceSec = 0.0d;
        double sampleCloseSec = 0.0d;
        String manifestPath;
        try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
                outDir.getAbsolutePath(),
                sampleMetaPath,
                schema,
                featureMetaPath,
                options)) {
            for (int sampleId = 0; sampleId < N_SAMPLES; sampleId++) {
                long sampleStart = System.nanoTime();
                ArraySampleParquetSampleContext sample = builder.sample(sampleId);
                try {
                    for (int featureId = 0; featureId < N_FEATURES; featureId++) {
                        long valueStart = System.nanoTime();
                        double[] value = valueForTrace(featureWaves[featureId], sampleOffsets[sampleId], featureBias[featureId]);
                        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
                        columns.put("time", time);
                        columns.put("value", value);
                        columns.put("ch_step", chStep);
                        valueGenerationSec += secondsSince(valueStart);

                        long addStart = System.nanoTime();
                        sample.addTrace(featureId, null, columns);
                        addTraceSec += secondsSince(addStart);
                    }
                } finally {
                    long closeStart = System.nanoTime();
                    sample.close();
                    sampleCloseSec += secondsSince(closeStart);
                }
                System.out.println("sample " + sampleId + " done in " + format(secondsSince(sampleStart)) + " sec");
            }
            long finishStart = System.nanoTime();
            manifestPath = builder.finish();
            double finishSec = secondsSince(finishStart);
            double buildSec = secondsSince(buildStart);
            printBuildResult(root, compression, metaSec, buildSec, valueGenerationSec, addTraceSec, sampleCloseSec, finishSec, manifestPath);
        }

        runQueries(manifestPath);
    }

    private static void printBuildResult(
            File root,
            String compression,
            double metaSec,
            double buildSec,
            double valueGenerationSec,
            double addTraceSec,
            double sampleCloseSec,
            double finishSec,
            String manifestPath) {
        File dataset = new File(root, "dataset");
        File sampleParts = new File(dataset, "sample_parts");
        File traceIndexParts = new File(dataset, "trace_index_parts");
        System.out.println("java_array_sample_parquet_benchmark");
        System.out.println("n_samples=" + N_SAMPLES);
        System.out.println("n_features=" + N_FEATURES);
        System.out.println("trace_len=" + TRACE_LEN);
        System.out.println("compression=" + compression);
        System.out.println("total_point_rows=" + ((long) N_SAMPLES * N_FEATURES * TRACE_LEN));
        System.out.println("meta_sec=" + format(metaSec));
        System.out.println("build_sec=" + format(buildSec));
        System.out.println("value_generation_sec=" + format(valueGenerationSec));
        System.out.println("add_trace_sec=" + format(addTraceSec));
        System.out.println("raw_sample_close_write_sec=" + format(sampleCloseSec));
        System.out.println("compact_finish_sec=" + format(finishSec));
        System.out.println("raw_samples_mb=" + format(bytesMb(sizeOf(new File(root, "dataset/raw_samples")))));
        System.out.println("raw_trace_index_mb=" + format(bytesMb(sizeOf(new File(root, "dataset/raw_trace_index")))));
        System.out.println("sample_parts_mb=" + format(bytesMb(sizeOf(sampleParts))));
        System.out.println("trace_index_parts_mb=" + format(bytesMb(sizeOf(traceIndexParts))));
        System.out.println("manifest=" + manifestPath);
    }

    private static void runQueries(String manifestPath) throws Exception {
        long[] allSamples = rangeLong(0, N_SAMPLES);
        int[] oneFeature = new int[]{123};
        long[] oneSample = new long[]{7L};
        long[] sixteenSamples = rangeLong(0, 16);
        int[] sixtyFourFeatures = rangeInt(0, 64);

        try (ArraySampleParquetReader reader = ArraySampleParquets.open(manifestPath)) {
            // 첫 반복은 DuckDB/JDBC 로딩과 metadata cache 초기화가 섞일 수 있어서 cold로 분리한다.
            QueryResult cold = measureQueries(reader, allSamples, oneFeature, oneSample, sixteenSamples, sixtyFourFeatures);
            QueryResult warm = measureQueries(reader, allSamples, oneFeature, oneSample, sixteenSamples, sixtyFourFeatures);
            System.out.println("query_cold_feature_all_samples_ms=" + format(cold.featureAllSamplesMs)
                    + " traces=" + cold.featureAllSamplesTraces);
            System.out.println("query_cold_sample_all_features_ms=" + format(cold.sampleAllFeaturesMs)
                    + " traces=" + cold.sampleAllFeaturesTraces);
            System.out.println("query_cold_16_samples_64_features_ms=" + format(cold.multiMs)
                    + " traces=" + cold.multiTraces);
            System.out.println("query_warm_feature_all_samples_ms=" + format(warm.featureAllSamplesMs)
                    + " traces=" + warm.featureAllSamplesTraces);
            System.out.println("query_warm_sample_all_features_ms=" + format(warm.sampleAllFeaturesMs)
                    + " traces=" + warm.sampleAllFeaturesTraces);
            System.out.println("query_warm_16_samples_64_features_ms=" + format(warm.multiMs)
                    + " traces=" + warm.multiTraces);
            System.out.println("query_materialized_points=" + warm.materializedPoints);
        }
    }

    private static QueryResult measureQueries(
            ArraySampleParquetReader reader,
            long[] allSamples,
            int[] oneFeature,
            long[] oneSample,
            long[] sixteenSamples,
            int[] sixtyFourFeatures) throws Exception {
        long q1 = System.nanoTime();
        List<ArraySampleParquetTrace> featureAllSamples = reader.loadTracesByIds(allSamples, oneFeature, false, true);
        double q1Ms = secondsSince(q1) * 1000.0d;

        long q2 = System.nanoTime();
        List<ArraySampleParquetTrace> sampleAllFeatures = reader.loadTracesByIds(oneSample, null, false, true);
        double q2Ms = secondsSince(q2) * 1000.0d;

        long q3 = System.nanoTime();
        List<ArraySampleParquetTrace> multi = reader.loadTracesByIds(sixteenSamples, sixtyFourFeatures, false, true);
        double q3Ms = secondsSince(q3) * 1000.0d;

        return new QueryResult(
                q1Ms,
                q2Ms,
                q3Ms,
                featureAllSamples.size(),
                sampleAllFeatures.size(),
                multi.size(),
                countPoints(featureAllSamples) + countPoints(sampleAllFeatures) + countPoints(multi));
    }

    private static final class QueryResult {
        final double featureAllSamplesMs;
        final double sampleAllFeaturesMs;
        final double multiMs;
        final int featureAllSamplesTraces;
        final int sampleAllFeaturesTraces;
        final int multiTraces;
        final long materializedPoints;

        QueryResult(
                double featureAllSamplesMs,
                double sampleAllFeaturesMs,
                double multiMs,
                int featureAllSamplesTraces,
                int sampleAllFeaturesTraces,
                int multiTraces,
                long materializedPoints) {
            this.featureAllSamplesMs = featureAllSamplesMs;
            this.sampleAllFeaturesMs = sampleAllFeaturesMs;
            this.multiMs = multiMs;
            this.featureAllSamplesTraces = featureAllSamplesTraces;
            this.sampleAllFeaturesTraces = sampleAllFeaturesTraces;
            this.multiTraces = multiTraces;
            this.materializedPoints = materializedPoints;
        }
    }

    private static long countPoints(List<ArraySampleParquetTrace> traces) {
        long total = 0L;
        for (ArraySampleParquetTrace trace : traces) {
            total += trace.traceLen;
        }
        return total;
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < N_SAMPLES; i++) {
            out.add(row("sample_key", String.format("sample_%06d", i), "split", (i % 5 == 0) ? "valid" : "train"));
        }
        return out;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < N_FEATURES; i++) {
            out.add(row("feature_key", String.format("feature_%06d", i), "family", "family_" + (i % 8)));
        }
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            row.put(kvs[i].toString(), kvs[i + 1]);
        }
        return row;
    }

    private static long[] rangeLong(int startInclusive, int endExclusive) {
        long[] out = new long[endExclusive - startInclusive];
        for (int i = 0; i < out.length; i++) {
            out[i] = startInclusive + i;
        }
        return out;
    }

    private static double[] linspace(double start, double end, int count) {
        double[] out = new double[count];
        if (count <= 1) {
            if (count == 1) {
                out[0] = start;
            }
            return out;
        }
        double step = (end - start) / (count - 1);
        for (int i = 0; i < count; i++) {
            out[i] = start + step * i;
        }
        return out;
    }

    private static double[] valueForTrace(double[] wave, double sampleOffset, double featureBias) {
        double[] out = new double[wave.length];
        double shift = sampleOffset + featureBias;
        for (int i = 0; i < wave.length; i++) {
            out[i] = wave[i] + shift;
        }
        return out;
    }

    private static int[] rangeInt(int startInclusive, int endExclusive) {
        int[] out = new int[endExclusive - startInclusive];
        for (int i = 0; i < out.length; i++) {
            out[i] = startInclusive + i;
        }
        return out;
    }

    private static double secondsSince(long startNs) {
        return (System.nanoTime() - startNs) / 1_000_000_000.0d;
    }

    private static String format(double value) {
        return String.format(java.util.Locale.US, "%.3f", value);
    }

    private static double bytesMb(long bytes) {
        return bytes / (1024.0d * 1024.0d);
    }

    private static long sizeOf(File path) {
        if (path == null || !path.exists()) {
            return 0L;
        }
        if (path.isFile()) {
            return path.length();
        }
        long total = 0L;
        File[] children = path.listFiles();
        if (children != null) {
            for (File child : children) {
                total += sizeOf(child);
            }
        }
        return total;
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
