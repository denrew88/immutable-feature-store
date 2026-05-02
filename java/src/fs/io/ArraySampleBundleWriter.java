package fs.io;

import fs.config.ArrayBundleConfig;
import fs.model.ArrayBundleManifest;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

public class ArraySampleBundleWriter implements AutoCloseable {
    private static final int INSERT_BATCH_SIZE = 256;

    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final int nSamples;
    private final ArrayBundleConfig config;
    private final File bundleDir;
    private final String manifestPath;
    private final Connection conn;

    private PreparedStatement insertPs;
    private int pendingBatch;
    private int currentRows;
    private long currentBytes;
    private int nBundles;
    private boolean finished;

    public ArraySampleBundleWriter(String outDir, String sampleMetaPath, int nSamples, ArrayBundleConfig config) throws Exception {
        this(outDir, sampleMetaPath, "", nSamples, config);
    }

    public ArraySampleBundleWriter(String outDir, String sampleMetaPath, String featureMetaPath, int nSamples, ArrayBundleConfig config) throws Exception {
        this.sampleMetaPath = (sampleMetaPath == null) ? "" : sampleMetaPath;
        this.featureMetaPath = (featureMetaPath == null) ? "" : featureMetaPath;
        this.nSamples = nSamples;
        this.config = (config == null) ? new ArrayBundleConfig() : config;
        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IOException("Failed to create out dir: " + out.getAbsolutePath());
        }
        this.bundleDir = new File(out, "array_sample_bundles");
        if (!bundleDir.exists() && !bundleDir.mkdirs()) {
            throw new IOException("Failed to create bundle dir: " + bundleDir.getAbsolutePath());
        }
        this.manifestPath = new File(out, "array_bundle_manifest.json").getAbsolutePath();
        this.conn = DuckDBUtils.connect(null);
        resetBundleTable();
    }

    public void appendTrace(long sampleId, int featureId, double[] time, double[] value) throws SQLException {
        ensureOpen();
        if (sampleId < 0 || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
        byte flags = ArrayFeatureFlags.compute(time, value);
        int traceLen = (time == null) ? 0 : time.length;
        byte[] timeBlob = ArrayUtils.encodeDoubleArray(time);
        byte[] valueBlob = ArrayUtils.encodeDoubleArray(value);

        insertPs.setLong(1, sampleId);
        insertPs.setInt(2, featureId);
        insertPs.setByte(3, flags);
        insertPs.setInt(4, traceLen);
        insertPs.setBytes(5, timeBlob);
        insertPs.setBytes(6, valueBlob);
        insertPs.addBatch();
        pendingBatch++;

        currentRows++;
        currentBytes += estimateRowBytes(traceLen);
        if (pendingBatch >= INSERT_BATCH_SIZE) {
            insertPs.executeBatch();
            pendingBatch = 0;
        }
        if (currentRows >= config.maxBundleRows || currentBytes >= config.maxBundleBytes) {
            flushBundle();
        }
    }

    public String finish() throws Exception {
        if (finished) {
            return manifestPath;
        }
        flushBundle();
        ArrayBundleManifest manifest = new ArrayBundleManifest(
                sampleMetaPath,
                featureMetaPath,
                nSamples,
                bundleDir.getAbsolutePath(),
                nBundles,
                "INT32",
                "UINT8",
                "FLOAT64_LE_BLOB",
                "FLOAT64_LE_BLOB");
        ArrayBundleManifestIO.write(manifest, manifestPath);
        finished = true;
        return manifestPath;
    }

    @Override
    public void close() throws Exception {
        try {
            finish();
        } finally {
            if (insertPs != null) {
                try {
                    insertPs.close();
                } catch (SQLException e) {
                    // best-effort close
                }
                insertPs = null;
            }
            conn.close();
        }
    }

    private void ensureOpen() {
        if (finished) {
            throw new IllegalStateException("writer already finished");
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
        String bundlePath = new File(bundleDir, String.format("bundle_%06d.parquet", nBundles)).getAbsolutePath();
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_array_bundle TO " + DuckDBUtils.quotePath(bundlePath) + " (FORMAT PARQUET)");
            st.execute("DELETE FROM tmp_array_bundle");
        }
        nBundles++;
        currentRows = 0;
        currentBytes = 0L;
    }

    private void resetBundleTable() throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_array_bundle ("
                    + "sample_id BIGINT, "
                    + "feature_id INTEGER, "
                    + "flags TINYINT, "
                    + "trace_len INTEGER, "
                    + "time_blob BLOB, "
                    + "value_blob BLOB)");
        }
        this.insertPs = conn.prepareStatement("INSERT INTO tmp_array_bundle VALUES (?, ?, ?, ?, ?, ?)");
        this.pendingBatch = 0;
    }

    private static long estimateRowBytes(int traceLen) {
        return 8L + 8L + 4L + 1L + 4L + ((long) traceLen * 16L);
    }
}
