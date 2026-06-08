package fs.io.scalar;

import fs.config.BuildShardConfig;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.DuckDBUtils;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarDenseLongPart;
import fs.model.scalar.ScalarSampleMajorManifest;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * scalar sample-major raw stage를 dense-long parquet shard로 materialize한다.
 *
 * <p>입력 parquet는 present 값만 가진 {@code (sample_id, feature_id, value)} row이다.
 * dense-long 최종 part는 모든 feature/sample 조합을 만들고, 입력 row가 없으면
 * {@code mask=0, value=NaN}으로 채운다. 이 방식은 파일 크기는 dense-long format
 * 표준 parquet 도구로 직접 디버깅하기 쉽고 sample 기준 batch 조회가 단순하다.
 */
public final class ScalarDenseLongShardBuilder {
    private static final int DEFAULT_ROW_GROUP_FEATURES = 128;
    private static final long DENSE_LONG_ROW_ESTIMATE_BYTES = 21L;
    private static final String DEFAULT_SCALAR_SHARD_MANIFEST_NAME = "scalar_shard_manifest.json";
    private static final String DEFAULT_SCALAR_SHARD_PARTS_DIR = "parts";

    private ScalarDenseLongShardBuilder() {
    }

    public static String buildFromSampleMajorManifest(String sampleMajorManifestPath, String outDir, BuildShardConfig config) throws Exception {
        BuildShardConfig cfg = config == null ? new BuildShardConfig() : config;
        ScalarSampleMajorManifest stage = ScalarSampleMajorManifestIO.read(sampleMajorManifestPath);
        File out = new File(outDir).getAbsoluteFile();
        File partsDir = new File(out, DEFAULT_SCALAR_SHARD_PARTS_DIR);
        File statsDir = new File(out, "selection_stats");
        if (!partsDir.exists() && !partsDir.mkdirs()) {
            throw new IOException("failed to create scalar shard parts dir: " + partsDir.getAbsolutePath());
        }
        if (!statsDir.exists() && !statsDir.mkdirs()) {
            throw new IOException("failed to create scalar shard selection stats dir: " + statsDir.getAbsolutePath());
        }

        String sampleMetaOut = new File(out, "sample_meta.parquet").getAbsolutePath();
        String featureMetaOut = new File(out, "feature_meta.parquet").getAbsolutePath();
        copyFile(stage.sampleMetaPath, sampleMetaOut);
        copyFile(stage.featureMetaPath, featureMetaOut);

        int nSamples = ArrayMetadataWriter.readRows(sampleMetaOut).size();
        int nFeatures = ArrayMetadataWriter.readRows(featureMetaOut).size();
        int rowGroupFeatures = cfg.denseLongRowGroupFeatures <= 0 ? DEFAULT_ROW_GROUP_FEATURES : cfg.denseLongRowGroupFeatures;
        int partFeatures = choosePartFeatures(nSamples, nFeatures, cfg, rowGroupFeatures);

        ArrayList<ScalarDenseLongPart> parts = new ArrayList<ScalarDenseLongPart>();
        LinkedHashMap<String, String> selectionStats = new LinkedHashMap<String, String>();

        try (Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP VIEW raw_values AS "
                    + "SELECT CAST(" + DuckDBUtils.quoteIdentifier(stage.sampleIdCol) + " AS BIGINT) AS sample_id, "
                    + "CAST(" + DuckDBUtils.quoteIdentifier(stage.featureIdCol) + " AS INTEGER) AS feature_id, "
                    + "CAST(" + DuckDBUtils.quoteIdentifier(stage.valueCol) + " AS DOUBLE) AS value "
                    + "FROM read_parquet(" + parquetList(stage.samplePaths) + ") "
                    + "WHERE " + DuckDBUtils.quoteIdentifier(stage.valueCol) + " IS NOT NULL "
                    + "AND NOT isnan(CAST(" + DuckDBUtils.quoteIdentifier(stage.valueCol) + " AS DOUBLE))");

            int partId = 0;
            for (int start = 0; start < nFeatures; start += partFeatures) {
                int end = Math.min(nFeatures, start + partFeatures);
                File partPath = new File(partsDir, String.format("part_%04d.parquet", Integer.valueOf(partId)));
                long rowCount = (long) (end - start) * (long) nSamples;
                copyDensePart(st, partPath.getAbsolutePath(), start, end, nSamples, rowGroupFeatures);
                parts.add(new ScalarDenseLongPart(
                        partId,
                        partPath.getAbsolutePath(),
                        start,
                        end - 1,
                        end - start,
                        rowCount,
                        partPath.length()));
                partId++;
            }

            String locatorPath = new File(out, "feature_locator.parquet").getAbsolutePath();
            copyFeatureLocator(st, locatorPath, nFeatures, nSamples, partFeatures);

            for (String yCol : resolveStatsYCols(cfg)) {
                String statsName = safeStatsName(yCol) + ".parquet";
                String statsPath = new File(statsDir, statsName).getAbsolutePath();
                copySelectionStats(st, statsPath, sampleMetaOut, yCol, nFeatures, partFeatures);
                selectionStats.put(yCol, statsPath);
            }

            String manifestPath = new File(out, DEFAULT_SCALAR_SHARD_MANIFEST_NAME).getAbsolutePath();
            ScalarDenseLongManifest manifest = new ScalarDenseLongManifest(
                    manifestPath,
                    sampleMetaOut,
                    featureMetaOut,
                    nSamples,
                    nFeatures,
                    partsDir.getAbsolutePath(),
                    parts,
                    locatorPath,
                    cfg.sampleKeyCol,
                    cfg.featureKeyCol,
                    cfg.sampleIdCol,
                    cfg.featureIdCol,
                    "value",
                    "mask",
                    "zstd",
                    rowGroupFeatures,
                    Long.valueOf(cfg.targetShardBytes),
                    selectionStats);
            ScalarDenseLongManifestIO.write(manifest, manifestPath);
            return manifestPath;
        }
    }

    private static void copyDensePart(
            Statement st,
            String path,
            int startFeatureId,
            int endFeatureId,
            int nSamples,
            int rowGroupFeatures) throws SQLException {
        String query =
                "WITH features AS (SELECT CAST(range AS INTEGER) AS feature_id FROM range("
                        + startFeatureId + ", " + endFeatureId + ")), "
                        + "samples AS (SELECT CAST(range AS BIGINT) AS sample_id FROM range(0, " + nSamples + ")), "
                        + "dense AS (SELECT f.feature_id, s.sample_id FROM features f CROSS JOIN samples s), "
                        + "sparse AS (SELECT feature_id, sample_id, value FROM raw_values "
                        + "WHERE feature_id >= " + startFeatureId + " AND feature_id < " + endFeatureId + ") "
                        + "SELECT d.feature_id, d.sample_id, "
                        + "CAST(CASE WHEN s.value IS NULL THEN 0 ELSE 1 END AS UTINYINT) AS mask, "
                        + "CASE WHEN s.value IS NULL THEN CAST('NaN' AS DOUBLE) ELSE CAST(s.value AS DOUBLE) END AS value "
                        + "FROM dense d LEFT JOIN sparse s USING (feature_id, sample_id) "
                        + "ORDER BY d.feature_id, d.sample_id";
        copyParquet(st, query, path, (long) nSamples * (long) rowGroupFeatures);
    }

    private static void copyFeatureLocator(
            Statement st,
            String path,
            int nFeatures,
            int nSamples,
            int partFeatures) throws SQLException {
        String query =
                "SELECT CAST(feature_id AS INTEGER) AS feature_id, "
                        + "CAST(feature_id AS INTEGER) AS global_rank, "
                        + "CAST(floor(feature_id / " + partFeatures + ") AS INTEGER) AS part_id, "
                        + "CAST(feature_id % " + partFeatures + " AS INTEGER) AS offset_in_part, "
                        + "CAST((feature_id % " + partFeatures + ") * " + nSamples + " AS BIGINT) AS first_row_in_part "
                        + "FROM range(0, " + nFeatures + ") AS t(feature_id) ORDER BY feature_id";
        copyParquet(st, query, path, 0L);
    }

    private static void copySelectionStats(
            Statement st,
            String path,
            String sampleMetaPath,
            String yCol,
            int nFeatures,
            int partFeatures) throws SQLException {
        String y = DuckDBUtils.quoteIdentifier(yCol);
        // n_y_overlap은 Y가 finite인 sample 수가 아니라
        // `(feature value가 finite로 존재함) AND (Y가 finite임)`의 교집합 크기다.
        // raw_values view가 이미 finite X만 포함하지만, stats SQL 자체도 같은 전제를
        // 명시해 두어 raw view 정의가 바뀌어도 Y-only count로 퇴화하지 않게 한다.
        String query =
                "WITH features AS (SELECT CAST(range AS INTEGER) AS feature_id FROM range(0, " + nFeatures + ")), "
                        + "ys AS (SELECT CAST(sample_id AS BIGINT) AS sample_id, CAST(" + y + " AS DOUBLE) AS y "
                        + "FROM read_parquet(" + DuckDBUtils.quotePath(sampleMetaPath) + ") "
                        + "WHERE " + y + " IS NOT NULL AND NOT isnan(CAST(" + y + " AS DOUBLE))), "
                        + "agg AS (SELECT x.feature_id, COUNT(*)::DOUBLE AS n, "
                        + "SUM(x.value)::DOUBLE AS sx, SUM(x.value * x.value)::DOUBLE AS sx2, "
                        + "SUM(ys.y)::DOUBLE AS sy, SUM(ys.y * ys.y)::DOUBLE AS sy2, "
                        + "SUM(x.value * ys.y)::DOUBLE AS sxy "
                        + "FROM raw_values x JOIN ys USING(sample_id) "
                        + "WHERE x.value IS NOT NULL AND NOT isnan(x.value) GROUP BY x.feature_id) "
                        + "SELECT f.feature_id, "
                        + "CAST(floor(f.feature_id / " + partFeatures + ") AS INTEGER) AS part_id, "
                        + "CAST(f.feature_id % " + partFeatures + " AS INTEGER) AS offset_in_part, "
                        + "CASE WHEN a.n > 1 AND (a.n * a.sx2 - a.sx * a.sx) > 0 AND (a.n * a.sy2 - a.sy * a.sy) > 0 "
                        + "THEN power(a.n * a.sxy - a.sx * a.sy, 2) / ((a.n * a.sx2 - a.sx * a.sx) * (a.n * a.sy2 - a.sy * a.sy)) "
                        + "ELSE 0.0 END AS r2y, "
                        + "CAST(COALESCE(a.n, 0) AS INTEGER) AS n_y_overlap "
                        + "FROM features f LEFT JOIN agg a USING(feature_id) ORDER BY f.feature_id";
        copyParquet(st, query, path, 0L);
    }

    private static void copyParquet(Statement st, String query, String path, long rowGroupSize) throws SQLException {
        String options = "(FORMAT PARQUET, COMPRESSION ZSTD";
        if (rowGroupSize > 0L) {
            options += ", ROW_GROUP_SIZE " + rowGroupSize;
        }
        options += ")";
        try {
            st.execute("COPY (" + query + ") TO " + DuckDBUtils.quotePath(path) + " " + options);
        } catch (SQLException first) {
            if (rowGroupSize <= 0L) {
                throw first;
            }
            st.execute("COPY (" + query + ") TO " + DuckDBUtils.quotePath(path) + " (FORMAT PARQUET, COMPRESSION ZSTD)");
        }
    }

    private static int choosePartFeatures(int nSamples, int nFeatures, BuildShardConfig cfg, int rowGroupFeatures) {
        if (cfg.denseLongPartFeatures > 0) {
            return Math.max(rowGroupFeatures, cfg.denseLongPartFeatures);
        }
        long perFeatureBytes = Math.max(1L, (long) nSamples * DENSE_LONG_ROW_ESTIMATE_BYTES);
        long estimated = Math.max(1L, cfg.targetShardBytes / perFeatureBytes);
        int partFeatures = (int) Math.min((long) nFeatures, Math.max((long) rowGroupFeatures, estimated));
        int remainder = partFeatures % rowGroupFeatures;
        if (remainder != 0) {
            partFeatures += rowGroupFeatures - remainder;
        }
        return Math.max(rowGroupFeatures, partFeatures);
    }

    private static List<String> resolveStatsYCols(BuildShardConfig cfg) {
        ArrayList<String> out = new ArrayList<String>();
        if (cfg.statsYCols != null) {
            for (String value : cfg.statsYCols) {
                if (value != null && !value.isEmpty() && !out.contains(value)) {
                    out.add(value);
                }
            }
        }
        if (out.isEmpty()) {
            out.add(cfg.yCol);
        }
        return out;
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

    private static String safeStatsName(String yCol) {
        String value = yCol == null || yCol.isEmpty() ? "y" : yCol;
        return value.replaceAll("[^A-Za-z0-9_.-]", "_");
    }

    private static void copyFile(String src, String dst) throws IOException {
        File dstFile = new File(dst);
        File parent = dstFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create metadata dir: " + parent.getAbsolutePath());
        }
        Files.copy(new File(src).toPath(), dstFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
}
