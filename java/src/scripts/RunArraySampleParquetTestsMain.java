package scripts;

import fs.io.ArraySampleParquets;
import fs.io.array_sample_parquet.ArraySampleParquetDatasetBuilder;
import fs.io.array_sample_parquet.ArraySampleParquetManifestIO;
import fs.io.array_sample_parquet.ArraySampleParquetOrderChecks;
import fs.io.array_sample_parquet.ArraySampleParquetSampleContext;
import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetBuildSessionStatus;
import fs.model.array_sample_parquet.ArraySampleParquetManifest;
import fs.model.array_sample_parquet.ArraySampleParquetTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.sql.Connection;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array_sample_parquet Java builder/reader 검증 스크립트다.
 */
public class RunArraySampleParquetTestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_array_sample_parquet_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = ArraySampleParquets.writeSampleMeta(sampleRows(), new File(root, "sample_meta.parquet").getAbsolutePath());
        String featureMetaPath = ArraySampleParquets.writeFeatureMeta(featureRows(), new File(root, "feature_meta.parquet").getAbsolutePath());
        List<PointColumnSpec> schema = Arrays.asList(
                new PointColumnSpec("ts", StorageType.INT64, LogicalType.TIMESTAMP_NS),
                new PointColumnSpec("value", StorageType.FLOAT64, LogicalType.CONTINUOUS),
                new PointColumnSpec("phase", StorageType.INT32, LogicalType.INTEGER),
                new PointColumnSpec("ch_step", StorageType.STRING, LogicalType.CATEGORICAL)
        );
        ArraySampleParquetBuildOptions options = new ArraySampleParquetBuildOptions();
        options.targetPartBytes = 1024L * 1024L;
        options.maxPartSamples = 2;
        options.compression = "none";

        File outDir = new File(root, "dataset");
        try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
                outDir.getAbsolutePath(),
                sampleMetaPath,
                schema,
                featureMetaPath,
                options)) {
            try (ArraySampleParquetSampleContext sample = builder.sample(0L)) {
                sample.addTrace(null, "feature_b", columns(
                        new long[]{3L},
                        new double[]{2.0},
                        new int[]{20},
                        new String[]{"B"}));
                sample.addTrace(null, "feature_a", columns(
                        new long[]{0L, 1L},
                        new double[]{1.0, Double.NaN},
                        new int[]{10, 11},
                        new String[]{"A", "B"}));
            }
            ArraySampleParquetOrderChecks.requirePointRowsSorted(rawSamplePath(outDir, 0L));
            ArraySampleParquetOrderChecks.requireTraceIndexRowsSorted(rawTraceIndexPath(outDir, 0L));
            try (ArraySampleParquetSampleContext ignored = builder.sample(1L)) {
                // empty sample checkpoint
            }
            ArraySampleParquetOrderChecks.requirePointRowsSorted(rawSamplePath(outDir, 1L));
            ArraySampleParquetOrderChecks.requireTraceIndexRowsSorted(rawTraceIndexPath(outDir, 1L));
            ArraySampleParquetBuildSessionStatus status = builder.status();
            require(status.nextExpectedSampleId == 2L, "expected resume sample 2");
            require(status.completedSampleCount == 2, "expected two completed raw samples");
            require(status.pendingSampleIds.size() == 2, "expected two pending raw samples");
        }

        String manifestPath;
        try (ArraySampleParquetDatasetBuilder builder = ArraySampleParquets.openSession(
                outDir.getAbsolutePath(),
                sampleMetaPath,
                schema,
                featureMetaPath,
                options)) {
            ArraySampleParquetBuildSessionStatus status = builder.status();
            require(status.nextExpectedSampleId == 2L, "resume status mismatch");
            require(status.pendingSampleIds.contains(Long.valueOf(2L)), "sample 2 should be pending");
            try (ArraySampleParquetSampleContext sample = builder.sample("sample_000002")) {
                sample.addTrace(null, "feature_b", columns(
                        new long[]{2L},
                        new double[]{3.0},
                        new int[]{12},
                        new String[]{"B"}));
            }
            try (ArraySampleParquetSampleContext sample = builder.sample(3L)) {
                sample.addTrace(null, "feature_a", columns(
                        new long[]{},
                        new double[]{},
                        new int[]{},
                        new String[]{}));
            }
            manifestPath = builder.finish();
        }

        require(new File(manifestPath).exists(), "missing manifest");
        ArraySampleParquetManifest manifest = ArraySampleParquetManifestIO.read(manifestPath);
        require(new File(manifest.parts.get(0).path).exists(), "missing point part");
        require(new File(manifest.parts.get(0).traceIndexPath).exists(), "missing trace index part");
        require(manifest.parts.get(0).rowCount == 3, "first point part should contain three point rows");
        ArraySampleParquetOrderChecks.requirePointRowsSorted(manifest.parts.get(0).path);
        ArraySampleParquetOrderChecks.requireTraceIndexRowsSorted(manifest.parts.get(0).traceIndexPath);
        assertOrderCheckDetectsUnsortedPointRows(root);
        try (fs.io.array_sample_parquet.ArraySampleParquetReader reader = ArraySampleParquets.open(manifestPath)) {
            List<ArraySampleParquetTrace> traces = reader.loadTracesByKeys(
                    new String[]{"sample_000000", "sample_000001", "sample_000002", "sample_000003"},
                    new String[]{"feature_a", "feature_b"},
                    true,
                    true);
            Map<String, ArraySampleParquetTrace> byPair = new LinkedHashMap<String, ArraySampleParquetTrace>();
            for (ArraySampleParquetTrace trace : traces) {
                byPair.put(trace.sampleId + ":" + trace.featureId, trace);
            }
            require(byPair.get("0:0").present, "feature_a sample0 should be present");
            assertStringArray((String[]) byPair.get("0:0").columns.get("ch_step"), new String[]{"A", "B"}, "sample0 ch_step");
            require(byPair.get("0:1").present, "feature_b sample0 should be present");
            assertStringArray((String[]) byPair.get("0:1").columns.get("ch_step"), new String[]{"B"}, "sample0 feature_b ch_step");
            require(!byPair.get("1:0").present, "sample1 feature_a should be missing");
            assertStringArray((String[]) byPair.get("2:1").columns.get("ch_step"), new String[]{"B"}, "sample2 ch_step");
            require(byPair.get("3:0").present, "sample3 feature_a should be present");
            require(byPair.get("3:0").traceLen == 0, "sample3 feature_a should be empty");
        }

        System.out.println("java array_sample_parquet tests passed");
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        out.add(row("sample_key", "sample_000000", "split", "train"));
        out.add(row("sample_key", "sample_000001", "split", "train"));
        out.add(row("sample_key", "sample_000002", "split", "valid"));
        out.add(row("sample_key", "sample_000003", "split", "test"));
        return out;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> out = new ArrayList<Map<String, Object>>();
        out.add(row("feature_key", "feature_a", "group", "alpha"));
        out.add(row("feature_key", "feature_b", "group", "beta"));
        return out;
    }

    private static LinkedHashMap<String, Object> columns(long[] ts, double[] value, int[] phase, String[] chStep) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        out.put("ts", ts);
        out.put("value", value);
        out.put("phase", phase);
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

    private static void assertStringArray(String[] actual, String[] expected, String label) {
        require(actual != null, label + " should not be null");
        require(actual.length == expected.length, label + " length mismatch");
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == null) {
                require(actual[i] == null, label + " mismatch at " + i);
            } else {
                require(expected[i].equals(actual[i]), label + " mismatch at " + i);
            }
        }
    }

    private static void assertOrderCheckDetectsUnsortedPointRows(File root) throws Exception {
        File path = new File(root, "unsorted_points.parquet");
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement()) {
            st.execute("COPY (SELECT * FROM (VALUES "
                    + "(0::BIGINT, 0::INTEGER, 1::INTEGER),"
                    + "(0::BIGINT, 0::INTEGER, 0::INTEGER)"
                    + ") AS t(sample_id, feature_id, point_idx)) TO "
                    + DuckDBUtils.quotePath(path.getAbsolutePath())
                    + " (FORMAT PARQUET, COMPRESSION 'uncompressed')");
        }
        require(!ArraySampleParquetOrderChecks.pointRowsSorted(path.getAbsolutePath()), "unsorted point rows should be detected");
    }

    private static String rawSamplePath(File outDir, long sampleId) {
        return new File(new File(outDir, "raw_samples"), String.format("sample_%012d.parquet", sampleId)).getAbsolutePath();
    }

    private static String rawTraceIndexPath(File outDir, long sampleId) {
        return new File(new File(outDir, "raw_trace_index"), String.format("sample_%012d.parquet", sampleId)).getAbsolutePath();
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
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
