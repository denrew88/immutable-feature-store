package fs.io;

import java.io.File;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

final class ScalarSampleBundleWriter implements AutoCloseable {
    private final File bundleDir;
    private final Connection conn;
    private final PreparedStatement insertPs;
    private final List<String> bundlePaths;
    private final int maxBundleRows;

    private int pendingBatch;
    private int bundleIndex;
    private int currentRows;
    private boolean finished;

    ScalarSampleBundleWriter(String outDir, int maxBundleRows) throws Exception {
        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IllegalStateException("failed to create sample-bundle dir: " + out.getAbsolutePath());
        }
        this.bundleDir = out;
        this.maxBundleRows = maxBundleRows;
        this.bundlePaths = new ArrayList<String>();
        this.conn = DuckDBUtils.connect(null);
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE tmp_scalar_sample_bundle (sample_id BIGINT, feature_id INTEGER, value DOUBLE)");
        }
        this.insertPs = conn.prepareStatement("INSERT INTO tmp_scalar_sample_bundle VALUES (?, ?, ?)");
        this.pendingBatch = 0;
        this.bundleIndex = 0;
        this.currentRows = 0;
        this.finished = false;
    }

    void appendSample(long sampleId, Map<Integer, Double> featureValues) throws SQLException {
        ensureOpen();
        if (featureValues == null || featureValues.isEmpty()) {
            return;
        }
        for (Map.Entry<Integer, Double> entry : featureValues.entrySet()) {
            Double value = entry.getValue();
            if (value == null || Double.isNaN(value.doubleValue())) {
                continue;
            }
            insertPs.setLong(1, sampleId);
            insertPs.setInt(2, entry.getKey().intValue());
            insertPs.setDouble(3, value.doubleValue());
            insertPs.addBatch();
            pendingBatch++;
            currentRows++;
            if (pendingBatch >= 1024) {
                insertPs.executeBatch();
                pendingBatch = 0;
            }
        }
        if (currentRows >= maxBundleRows) {
            flushBundle();
        }
    }

    List<String> finish() throws Exception {
        if (finished) {
            return new ArrayList<String>(bundlePaths);
        }
        flushBundle();
        finished = true;
        return new ArrayList<String>(bundlePaths);
    }

    @Override
    public void close() throws Exception {
        try {
            finish();
        } finally {
            try {
                insertPs.close();
            } finally {
                conn.close();
            }
        }
    }

    private void flushBundle() throws SQLException {
        if (pendingBatch > 0) {
            insertPs.executeBatch();
            pendingBatch = 0;
        }
        if (currentRows == 0) {
            return;
        }
        String bundlePath = new File(bundleDir, String.format("bundle_%06d.parquet", bundleIndex)).getAbsolutePath();
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_scalar_sample_bundle TO " + DuckDBUtils.quotePath(bundlePath) + " (FORMAT PARQUET)");
            st.execute("DELETE FROM tmp_scalar_sample_bundle");
        }
        bundlePaths.add(bundlePath);
        bundleIndex++;
        currentRows = 0;
    }

    private void ensureOpen() {
        if (finished) {
            throw new IllegalStateException("scalar sample bundle writer is already finished");
        }
    }
}
