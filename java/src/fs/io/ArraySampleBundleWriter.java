package fs.io;

import fs.config.ArrayBundleConfig;
import fs.model.ArrayBundleManifest;
import fs.model.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ArraySampleBundleWriter implements AutoCloseable {
    private static final int INSERT_BATCH_SIZE = 256;

    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final int nSamples;
    private final ArrayBundleConfig config;
    private final File bundleDir;
    private final String manifestPath;
    private final Connection conn;
    private final List<PointColumnSpec> pointSchema;

    private PreparedStatement insertPs;
    private String insertSql;
    private int pendingBatch;
    private int currentRows;
    private long currentBytes;
    private int nBundles;
    private boolean finished;

    public ArraySampleBundleWriter(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            ArrayBundleConfig config,
            List<PointColumnSpec> pointSchema) throws Exception {
        this.sampleMetaPath = (sampleMetaPath == null) ? "" : sampleMetaPath;
        this.featureMetaPath = (featureMetaPath == null) ? "" : featureMetaPath;
        this.nSamples = nSamples;
        this.config = (config == null) ? new ArrayBundleConfig() : config;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
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

    public void appendTrace(long sampleId, int featureId, Map<String, Object> columns) throws SQLException {
        ensureOpen();
        if (sampleId < 0 || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
        LinkedHashMap<String, Object> normalized = normalizeColumns(columns);
        byte flags = ArrayFeatureFlags.compute(normalized);
        int traceLen = 0;
        int paramIndex = 1;
        insertPs.setLong(1, sampleId);
        insertPs.setInt(2, featureId);
        insertPs.setByte(3, flags);
        for (PointColumnSpec spec : pointSchema) {
            int columnTraceLen = ArrayUtils.pointColumnLength(normalized.get(spec.name));
            if (traceLen == 0) {
                traceLen = columnTraceLen;
            } else if (traceLen != columnTraceLen) {
                throw new IllegalArgumentException("point column length mismatch for " + spec.name);
            }
        }
        insertPs.setInt(4, traceLen);
        paramIndex = 5;
        for (PointColumnSpec spec : pointSchema) {
            byte[] blob = ArrayUtils.encodePointColumn(normalized.get(spec.name), spec);
            insertPs.setBytes(paramIndex++, blob);
        }
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
                pointSchema);
        ArrayBundleManifestIO.write(manifest, manifestPath);
        finished = true;
        return manifestPath;
    }

    public void updatePointSchema(List<PointColumnSpec> pointSchema) {
        List<PointColumnSpec> normalized = PointColumnSpec.normalizeList(pointSchema);
        if (normalized.size() != this.pointSchema.size()) {
            throw new IllegalArgumentException("point_schema column count cannot change after bundle writer initialization");
        }
        this.pointSchema.clear();
        this.pointSchema.addAll(normalized);
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
        StringBuilder create = new StringBuilder();
        create.append("CREATE TEMP TABLE IF NOT EXISTS tmp_array_bundle (");
        create.append("sample_id BIGINT, ");
        create.append("feature_id INTEGER, ");
        create.append("flags TINYINT, ");
        create.append("trace_len INTEGER");
        for (PointColumnSpec spec : pointSchema) {
            create.append(", ").append(blobColumnName(spec.name)).append(" BLOB");
        }
        create.append(")");
        StringBuilder insert = new StringBuilder();
        insert.append("INSERT INTO tmp_array_bundle VALUES (?, ?, ?, ?");
        for (int i = 0; i < pointSchema.size(); i++) {
            insert.append(", ?");
        }
        insert.append(")");
        try (Statement st = conn.createStatement()) {
            st.execute(create.toString());
        }
        this.insertSql = insert.toString();
        this.insertPs = conn.prepareStatement(insertSql);
        this.pendingBatch = 0;
    }

    private long estimateRowBytes(int traceLen) {
        long bytes = 8L + 8L + 4L + 1L + 4L;
        for (PointColumnSpec spec : pointSchema) {
            bytes += (long) traceLen * (long) spec.storageType.itemSize;
        }
        return bytes;
    }

    private boolean hasColumn(String name) {
        for (PointColumnSpec spec : pointSchema) {
            if (spec.name.equals(name)) {
                return true;
            }
        }
        return false;
    }

    private LinkedHashMap<String, Object> normalizeColumns(Map<String, Object> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("columns must not be null");
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (Map.Entry<String, Object> entry : columns.entrySet()) {
            out.put(entry.getKey(), entry.getValue());
        }
        for (PointColumnSpec spec : pointSchema) {
            if (!out.containsKey(spec.name)) {
                throw new IllegalArgumentException("missing point column: " + spec.name);
            }
        }
        if (out.size() != pointSchema.size()) {
            for (String name : out.keySet()) {
                if (!hasColumn(name)) {
                    throw new IllegalArgumentException("unexpected point column: " + name);
                }
            }
        }
        return out;
    }

    private static String blobColumnName(String name) {
        return name + "_blob";
    }
}
