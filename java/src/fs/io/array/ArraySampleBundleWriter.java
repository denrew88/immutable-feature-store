package fs.io.array;

import fs.config.ArrayBundleConfig;
import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.model.array.ArrayBundleManifest;
import fs.model.common.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array trace row를 sample-major bundle parquet로 쌓는 writer다.
 *
 * <p>direct-ingestion builder는 trace를 하나씩 이 writer에 넣고, 이 writer는
 * 일정 row 수나 추정 바이트 수를 넘으면 임시 DuckDB 테이블을 parquet bundle 하나로 flush한다.
 * 최종 shard build는 이 bundle 목록을 다시 읽어 binary shard로 변환한다.
 */
public class ArraySampleBundleWriter implements AutoCloseable {
    private static final int INSERT_BATCH_SIZE = 256;
    private static final long SAMPLE_ID_BYTES = 8L;
    // Keep the historical row-size estimate stable so bundle flush behavior does not shift.
    private static final long ROW_ENCODING_OVERHEAD_BYTES = 8L;
    private static final long FEATURE_ID_BYTES = 4L;
    private static final long FLAGS_BYTES = 1L;
    private static final long TRACE_LEN_BYTES = 4L;
    private static final long BUNDLE_TRACE_ROW_FIXED_BYTES =
            SAMPLE_ID_BYTES + ROW_ENCODING_OVERHEAD_BYTES + FEATURE_ID_BYTES + FLAGS_BYTES + TRACE_LEN_BYTES;

    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final int nSamples;
    private final ArrayBundleConfig config;
    private final File bundleDir;
    private final String manifestPath;
    private final Connection conn;
    private final List<PointColumnSpec> pointSchema;
    private final boolean autoFlush;

    private PreparedStatement insertPs;
    private String insertSql;
    private int pendingBatch;
    private int currentRows;
    private long currentBytes;
    private int nBundles;
    private boolean finished;

    /**
     * seal된 bundle parquet 하나의 commit 결과를 나타낸다.
     */
    public static final class BundleCommit {
        public final int bundleId;
        public final String path;
        public final int rowCount;
        public final long byteSize;

        BundleCommit(int bundleId, String path, int rowCount, long byteSize) {
            this.bundleId = bundleId;
            this.path = path;
            this.rowCount = rowCount;
            this.byteSize = byteSize;
        }
    }

    /**
     * bundle writer를 초기화한다.
     *
     * @param outDir bundle stage 루트 디렉터리
     * @param sampleMetaPath sample metadata parquet 경로
     * @param featureMetaPath feature metadata parquet 경로
     * @param nSamples dense sample 개수
     * @param config bundle flush 기준
     * @param pointSchema point column schema
     */
    public ArraySampleBundleWriter(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            ArrayBundleConfig config,
            List<PointColumnSpec> pointSchema) throws Exception {
        this(outDir, sampleMetaPath, featureMetaPath, nSamples, config, pointSchema, 0, true);
    }

    /**
     * resume-safe builder가 bundle 번호를 이어가거나 auto flush를 직접 제어할 수 있게 하는 생성자다.
     */
    public ArraySampleBundleWriter(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            ArrayBundleConfig config,
            List<PointColumnSpec> pointSchema,
            int startBundleId,
            boolean autoFlush) throws Exception {
        this.sampleMetaPath = (sampleMetaPath == null) ? "" : sampleMetaPath;
        this.featureMetaPath = (featureMetaPath == null) ? "" : featureMetaPath;
        this.nSamples = nSamples;
        this.config = (config == null) ? new ArrayBundleConfig() : config;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        this.autoFlush = autoFlush;
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
        this.nBundles = startBundleId;
        resetBundleTable();
    }

    /**
     * trace 하나를 현재 bundle 버퍼에 추가한다.
     *
     * <p>이 메서드는 point column 길이를 검증하고 각 column을 blob으로 인코딩한 뒤
     * 임시 DuckDB 테이블에 적재한다. flush 기준은 row 수와 추정 직렬화 바이트 수를 함께 본다.
     *
     * @param sampleId dense sample id
     * @param featureId dense feature id
     * @param columns schema 순서와 이름이 맞는 point column 값
     */
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
        if (autoFlush && shouldFlushBundle()) {
            flushBundle();
        }
    }

    /**
     * 현재 bundle 버퍼가 row/byte 기준을 넘어 flush 후보인지 알려준다.
     */
    public boolean shouldFlushBundle() {
        return currentRows >= config.maxBundleRows || currentBytes >= config.maxBundleBytes;
    }

    /**
     * 다음에 기록될 bundle id를 돌려준다.
     */
    public int nextBundleId() {
        return nBundles;
    }

    /**
     * 남은 버퍼를 flush하고 bundle manifest를 작성한다.
     *
     * @return 생성된 bundle manifest 경로
     */
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

    /**
     * 최종 manifest에 기록할 point schema를 갱신한다.
     *
     * <p>categorical dictionary 경로처럼 build 과정에서 뒤늦게 확정되는 정보만 바뀌어야 하므로
     * 컬럼 개수 자체는 바뀌지 않도록 막는다.
     */
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

    /**
     * 현재 임시 테이블 내용을 parquet bundle 하나로 materialize한다.
     */
    public BundleCommit flushBundle() throws SQLException {
        if (pendingBatch > 0) {
            insertPs.executeBatch();
            pendingBatch = 0;
        }
        if (currentRows == 0) {
            return null;
        }
        int bundleId = nBundles;
        String bundlePath = new File(bundleDir, String.format("bundle_%06d.parquet", bundleId)).getAbsolutePath();
        String tmpBundlePath = bundlePath + ".tmp";
        int rowCount = currentRows;
        long byteSize = currentBytes;
        try (Statement st = conn.createStatement()) {
            st.execute("COPY tmp_array_bundle TO " + DuckDBUtils.quotePath(tmpBundlePath) + " (FORMAT PARQUET)");
            st.execute("DELETE FROM tmp_array_bundle");
        }
        try {
            java.nio.file.Files.move(
                    new File(tmpBundlePath).toPath(),
                    new File(bundlePath).toPath(),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new SQLException("failed to finalize bundle parquet: " + bundlePath, e);
        }
        nBundles++;
        currentRows = 0;
        currentBytes = 0L;
        return new BundleCommit(bundleId, bundlePath, rowCount, byteSize);
    }

    /**
     * bundle row를 적재할 임시 DuckDB 테이블과 insert statement를 준비한다.
     */
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

    /**
     * trace 한 행이 bundle 안에서 차지할 대략적인 직렬화 바이트 수를 추정한다.
     *
     * <p>고정 오버헤드는 sample_id, feature_id, flags, trace_len 같은 메타데이터 필드이고,
     * 가변 크기 부분은 trace 길이와 각 point column 원소 크기를 곱해 계산한다.
     */
    private long estimateRowBytes(int traceLen) {
        long bytes = BUNDLE_TRACE_ROW_FIXED_BYTES;
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

    /**
     * 입력 column map이 point schema와 정확히 일치하는지 검증한다.
     */
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
