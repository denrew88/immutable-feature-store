package fs.io;

import fs.io.common.DuckDBUtils;
import fs.io.scalar.FeatureIdIndex;
import fs.io.scalar.SampleIdIndex;
import fs.io.scalar.ScalarDenseLongManifestIO;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarDenseLongPart;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;
import fs.model.selection.Candidate;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * dense-long scalar parquet shard reader다.
 *
 * <p>이 reader는 표준 parquet rows를 DuckDB로 필터링해서 읽는다. feature 기준 조회는
 * manifest의 part range를 먼저 사용해 대상 part를 좁히고, sample 기준 조회는 모든 part에
 * {@code sample_id} filter를 걸어 batch scan한다.
 */
public final class ScalarDenseLongDataset implements AutoCloseable {
    private final ScalarDenseLongManifest manifest;
    private final Connection conn;
    private final SampleIdIndex sampleIndex;
    private final FeatureIdIndex featureIndex;

    public ScalarDenseLongDataset(String manifestPath) throws Exception {
        this.manifest = ScalarDenseLongManifestIO.read(manifestPath);
        this.conn = DuckDBUtils.connect(null);
        this.sampleIndex = SampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
        this.featureIndex = FeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
    }

    public ScalarDenseLongManifest manifest() {
        return manifest;
    }

    public ScalarFeatureValues loadFeatureById(int featureId) throws Exception {
        ScalarDenseLongPart part = partForFeature(featureId);
        ArrayList<ScalarValue> values = new ArrayList<ScalarValue>(manifest.nSamples);
        Double[] rawValues = new Double[manifest.nSamples];
        boolean[] present = new boolean[manifest.nSamples];
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT sample_id, mask, value FROM read_parquet(" + DuckDBUtils.quotePath(part.path) + ") "
                             + "WHERE feature_id = " + featureId + " ORDER BY sample_id")) {
            while (rs.next()) {
                int sampleId = (int) rs.getLong(1);
                boolean isPresent = rs.getInt(2) != 0;
                present[sampleId] = isPresent;
                rawValues[sampleId] = isPresent ? Double.valueOf(rs.getDouble(3)) : null;
            }
        }
        for (int sampleId = 0; sampleId < manifest.nSamples; sampleId++) {
            values.add(new ScalarValue(
                    sampleId,
                    sampleIndex.keyForId(sampleId),
                    present[sampleId],
                    rawValues[sampleId]));
        }
        return new ScalarFeatureValues(featureId, featureIndex.keyForId(featureId), values);
    }

    public ScalarFeatureValues loadFeatureByKey(String featureKey) throws Exception {
        Integer featureId = featureIndex.findFeatureIdByKey(featureKey);
        if (featureId == null) {
            throw new IllegalArgumentException("unknown feature key: " + featureKey);
        }
        return loadFeatureById(featureId.intValue());
    }

    public SampleValues loadSampleById(long sampleId) throws Exception {
        if (sampleId < 0L || sampleId >= manifest.nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
        double[] values = new double[manifest.nFeatures];
        byte[] valid = new byte[manifest.nFeatures];
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT feature_id, mask, value FROM read_parquet(" + parquetList(partPaths()) + ") "
                             + "WHERE sample_id = " + sampleId + " ORDER BY feature_id")) {
            while (rs.next()) {
                int featureId = rs.getInt(1);
                int mask = rs.getInt(2);
                valid[featureId] = (byte) mask;
                values[featureId] = rs.getDouble(3);
            }
        }
        return new SampleValues(sampleId, sampleIndex.keyForId(sampleId), values, valid);
    }

    public SampleValues loadSampleByKey(String sampleKey) throws Exception {
        Long sampleId = sampleIndex.findSampleIdByKey(sampleKey);
        if (sampleId == null) {
            throw new IllegalArgumentException("unknown sample key: " + sampleKey);
        }
        return loadSampleById(sampleId.longValue());
    }

    public List<Candidate> topFeaturesFromStats(String yCol, int topK) throws Exception {
        String path = manifest.selectionStatsPath(yCol == null || yCol.isEmpty() ? "y" : yCol);
        if (path.isEmpty()) {
            throw new IllegalArgumentException("selection stats not found for y column: " + yCol);
        }
        ArrayList<Candidate> out = new ArrayList<Candidate>();
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT feature_id, part_id, offset_in_part, r2y, n_y_overlap "
                             + "FROM read_parquet(" + DuckDBUtils.quotePath(path) + ") "
                             + "ORDER BY r2y DESC NULLS LAST, feature_id ASC LIMIT " + topK)) {
            while (rs.next()) {
                out.add(new Candidate(rs.getInt(1), rs.getInt(2), rs.getInt(3), rs.getDouble(4), rs.getInt(5)));
            }
        }
        return out;
    }

    @Override
    public void close() throws Exception {
        conn.close();
    }

    private ScalarDenseLongPart partForFeature(int featureId) {
        if (featureId < 0 || featureId >= manifest.nFeatures) {
            throw new IllegalArgumentException("feature_id out of range: " + featureId);
        }
        for (ScalarDenseLongPart part : manifest.parts) {
            if (featureId >= part.firstFeatureId && featureId <= part.lastFeatureId) {
                return part;
            }
        }
        throw new IllegalArgumentException("feature_id not covered by dense-long parts: " + featureId);
    }

    private List<String> partPaths() {
        ArrayList<String> paths = new ArrayList<String>(manifest.parts.size());
        for (ScalarDenseLongPart part : manifest.parts) {
            paths.add(part.path);
        }
        return paths;
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

    /**
     * sample 하나에 대한 전체 feature dense vector다.
     */
    public static final class SampleValues {
        public final long sampleId;
        public final String sampleKey;
        public final double[] values;
        public final byte[] valid;

        public SampleValues(long sampleId, String sampleKey, double[] values, byte[] valid) {
            this.sampleId = sampleId;
            this.sampleKey = sampleKey;
            this.values = Arrays.copyOf(values, values.length);
            this.valid = Arrays.copyOf(valid, valid.length);
        }
    }
}
