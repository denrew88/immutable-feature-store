package scripts;

import fs.config.BuildShardConfig;
import fs.io.common.DuckDBUtils;
import fs.io.ScalarDatasetBuilder;
import fs.io.ScalarDenseLongDataset;
import fs.io.ScalarFeatureShards;
import fs.model.scalar.ScalarBuildSessionStatus;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarDenseLongPart;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;

import java.io.File;
import java.io.FileOutputStream;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Scalar standard builder, reader, selection facade를 함께 검증하는 테스트 entrypoint.
 */
public class RunScalarBuilderTestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_scalar_builder_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        String sampleMetaPath = new File(root, "sample_meta.parquet").getAbsolutePath();
        String featureMetaPath = new File(root, "feature_meta.parquet").getAbsolutePath();
        ScalarFeatureShards.writeSampleMeta(sampleRows(), sampleMetaPath);
        ScalarFeatureShards.writeFeatureMeta(featureRows(), featureMetaPath);

        BuildShardConfig cfg = new BuildShardConfig();
        cfg.targetShardBytes = 1L << 20;
        cfg.statsYCols = java.util.Arrays.asList("y", "y_alt", "y_const");
        cfg.denseLongRowGroupFeatures = 128;
        cfg.denseLongPartFeatures = 128;

        File outDir = new File(root, "scalar_standard");
        String manifestPath;
        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir.getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            ScalarBuildSessionStatus status = builder.status();
            require(status.pendingSampleIds.equals(java.util.Arrays.asList(0L, 1L, 2L, 3L)), "initial pending ids mismatch");
            File staleLegacyTmp = new File(new File(outDir, "raw_samples"), "sample_000000000002.parquet.tmp");
            try (FileOutputStream staleHandle = new FileOutputStream(staleLegacyTmp)) {
                staleHandle.write(1);
                staleHandle.flush();
                builder.writeSample(2L, values("feature_01", 2.0), true);
            }
            if (staleLegacyTmp.exists() && !staleLegacyTmp.delete()) {
                staleLegacyTmp.deleteOnExit();
            }
            builder.writeSample(0L, values("feature_01", 1.0, "feature_04", 4.0), true);
            ScalarBuildSessionStatus mid = builder.status();
            require(mid.completedSampleCount == 2, "completed sample count mismatch");
            require(mid.pendingSampleIds.equals(java.util.Arrays.asList(1L, 3L)), "pending ids mismatch after first run");
        }

        try (ScalarDatasetBuilder builder = ScalarFeatureShards.openSession(
                outDir.getAbsolutePath(),
                sampleMetaPath,
                featureMetaPath,
                null,
                cfg)) {
            builder.writeSample(3L, values("feature_03", 3.0), true);
            builder.writeSample(1L, values("feature_01", 1.5, "feature_05", 5.0), true);
            String stageManifest = builder.finishStage();
            require(new File(stageManifest).exists(), "sample-major manifest missing");
            manifestPath = builder.buildShards(true, new File(root, "scalar_shard").getAbsolutePath());
        }

        ScalarDenseLongManifest manifest = ScalarFeatureShards.loadManifest(manifestPath);
        require(manifest.nSamples == 4, "n_samples mismatch");
        require(manifest.nFeatures == 6, "n_features mismatch");
        require(new File(manifest.featureMetaPath).exists(), "missing feature_meta.parquet");
        require(new File(manifest.sampleMetaPath).exists(), "missing sample_meta.parquet");
        require(!manifest.selectionStats.isEmpty(), "selection_stats should be populated");
        assertStatsOverlapMatchesFinalMask(manifest, "y");
        assertStatsOverlapMatchesFinalMask(manifest, "y_alt");
        assertStatsOverlapMatchesFinalMask(manifest, "y_const");
        assertStatsOverlap(manifest, "y_alt", 1, 2);
        assertStatsOverlap(manifest, "y_const", 1, 3);
        assertStatsR2Zero(manifest, "y_const");

        try (ScalarDenseLongDataset dataset = ScalarFeatureShards.openDenseLong(manifestPath)) {
            ScalarFeatureValues feature01 = dataset.loadFeatureByKey("feature_01");
            require(feature01.values.size() == 4, "feature sample count mismatch");
            assertValue(feature01.values.get(0), 0L, "sample_000000", true, 1.0);
            assertValue(feature01.values.get(1), 1L, "sample_000001", true, 1.5);
            assertValue(feature01.values.get(2), 2L, "sample_000002", true, 2.0);
            assertValue(feature01.values.get(3), 3L, "sample_000003", false, null);

            ScalarDenseLongDataset.SampleValues sample0 = dataset.loadSampleById(0L);
            require(sample0.valid[1] == 1, "sample value should be present");
            require(Math.abs(sample0.values[1] - 1.0) <= 1e-12, "sample value mismatch");
            require(sample0.valid[2] == 0, "missing value should have mask=0");
            require(Double.isNaN(sample0.values[2]), "missing value should be stored as NaN");
            require(!dataset.topFeaturesFromStats("y", 2).isEmpty(), "selection stats should produce candidates");
        }

        System.out.println("java scalar builder tests passed");
    }

    private static List<Map<String, Object>> sampleRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        rows.add(row("sample_key", "sample_000000", "y", 0.0, "y_alt", 1.0, "y_const", 7.0, "split", "train"));
        rows.add(row("sample_key", "sample_000001", "y", 1.0, "y_alt", 0.0, "y_const", 7.0, "split", "train"));
        rows.add(row("sample_key", "sample_000002", "y", 0.0, "y_alt", Double.NaN, "y_const", 7.0, "split", "test"));
        rows.add(row("sample_key", "sample_000003", "y", 1.0, "y_alt", 0.0, "y_const", 7.0, "split", "test"));
        return rows;
    }

    private static List<Map<String, Object>> featureRows() {
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < 6; i++) {
            rows.add(row("feature_key", String.format("feature_%02d", Integer.valueOf(i)), "group", "ordered"));
        }
        return rows;
    }

    private static LinkedHashMap<Object, Object> values(Object... kvs) {
        LinkedHashMap<Object, Object> out = new LinkedHashMap<Object, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i], kvs[i + 1]);
        }
        return out;
    }

    private static LinkedHashMap<String, Object> row(Object... kvs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kvs.length; i += 2) {
            out.put(kvs[i].toString(), kvs[i + 1]);
        }
        return out;
    }

    private static void assertValue(ScalarValue actual, long sampleId, String sampleKey, boolean present, Double value) {
        require(actual != null, "value result should not be null");
        require(actual.sampleId == sampleId, "sample_id mismatch");
        require(sampleKey.equals(actual.sampleKey), "sample_key mismatch");
        require(actual.present == present, "present flag mismatch for sample_id=" + sampleId);
        if (!present) {
            require(actual.value == null, "missing value should be null");
            return;
        }
        require(actual.value != null, "present value should not be null");
        require(Math.abs(actual.value.doubleValue() - value.doubleValue()) <= 1e-12, "value mismatch for sample_id=" + sampleId);
    }

    private static void assertStatsOverlap(ScalarDenseLongManifest manifest, String yCol, int featureId, int expected) throws Exception {
        String statsPath = manifest.selectionStatsPath(yCol);
        require(statsPath != null && !statsPath.isEmpty(), "missing selection stats for " + yCol);
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT n_y_overlap FROM read_parquet(" + DuckDBUtils.quotePath(statsPath) + ") "
                             + "WHERE feature_id = " + featureId)) {
            require(rs.next(), "missing stats row for feature_id=" + featureId);
            int actual = rs.getInt(1);
            require(actual == expected, "n_y_overlap mismatch for " + yCol + "/feature_id=" + featureId
                    + ": actual=" + actual + " expected=" + expected);
        }
    }

    private static void assertStatsOverlapMatchesFinalMask(ScalarDenseLongManifest manifest, String yCol) throws Exception {
        String statsPath = manifest.selectionStatsPath(yCol);
        require(statsPath != null && !statsPath.isEmpty(), "missing selection stats for " + yCol);
        ArrayList<String> partPaths = new ArrayList<String>();
        for (ScalarDenseLongPart part : manifest.parts) {
            partPaths.add(part.path);
        }
        require(!partPaths.isEmpty(), "scalar parts should not be empty");
        String y = DuckDBUtils.quoteIdentifier(yCol);
        String sql =
                "WITH features AS (SELECT CAST(range AS INTEGER) AS feature_id FROM range(0, " + manifest.nFeatures + ")), "
                        + "final_mask AS (SELECT CAST(feature_id AS INTEGER) AS feature_id, CAST(sample_id AS BIGINT) AS sample_id "
                        + "FROM read_parquet(" + parquetList(partPaths) + ") WHERE CAST(mask AS INTEGER) = 1), "
                        + "ys AS (SELECT CAST(sample_id AS BIGINT) AS sample_id FROM read_parquet(" + DuckDBUtils.quotePath(manifest.sampleMetaPath) + ") "
                        + "WHERE " + y + " IS NOT NULL AND NOT isnan(CAST(" + y + " AS DOUBLE))), "
                        + "expected AS (SELECT f.feature_id, CAST(COUNT(ys.sample_id) AS INTEGER) AS expected_n "
                        + "FROM features f LEFT JOIN final_mask x ON x.feature_id = f.feature_id "
                        + "LEFT JOIN ys ON ys.sample_id = x.sample_id GROUP BY f.feature_id), "
                        + "stats AS (SELECT CAST(feature_id AS INTEGER) AS feature_id, CAST(n_y_overlap AS INTEGER) AS n_y_overlap "
                        + "FROM read_parquet(" + DuckDBUtils.quotePath(statsPath) + ")), "
                        + "valid_y AS (SELECT COUNT(*) AS y_count FROM ys) "
                        + "SELECT "
                        + "(SELECT COUNT(*) FROM stats s JOIN expected e USING(feature_id) WHERE s.n_y_overlap <> e.expected_n) AS mismatch_count, "
                        + "(SELECT COUNT(*) FROM expected, valid_y WHERE expected.expected_n < valid_y.y_count) AS feature_limited_count";
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            require(rs.next(), "stats overlap direct check did not return a row for " + yCol);
            long mismatchCount = rs.getLong(1);
            long featureLimitedCount = rs.getLong(2);
            require(mismatchCount == 0L, "n_y_overlap should equal count(mask=1 AND finite " + yCol + "); mismatch_count=" + mismatchCount);
            require(featureLimitedCount > 0L, "test fixture does not prove " + yCol + " stats are feature-limited, not Y-only");
        }
    }

    private static void assertStatsR2Zero(ScalarDenseLongManifest manifest, String yCol) throws Exception {
        String statsPath = manifest.selectionStatsPath(yCol);
        require(statsPath != null && !statsPath.isEmpty(), "missing selection stats for " + yCol);
        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT COUNT(*) FROM read_parquet(" + DuckDBUtils.quotePath(statsPath) + ") "
                             + "WHERE COALESCE(r2y, 0.0) <> 0.0")) {
            require(rs.next(), "missing r2 count result for " + yCol);
            require(rs.getLong(1) == 0L, "constant y should produce r2y=0.0 for " + yCol);
        }
    }

    private static String parquetList(List<String> paths) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < paths.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(DuckDBUtils.quotePath(paths.get(i)));
        }
        sb.append("]");
        return sb.toString();
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
