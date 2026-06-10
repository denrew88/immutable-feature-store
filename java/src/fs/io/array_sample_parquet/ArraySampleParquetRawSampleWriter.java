package fs.io.array_sample_parquet;

import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

import java.io.File;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * sample 하나를 raw point parquet와 raw trace-index parquet로 저장하는 writer입니다.
 *
 * <p>이 writer의 중요한 제약은 sample-feature 쌍 하나가 trace 하나만 가질 수 있다는 점입니다.
 * 같은 sample 안에서 같은 feature에 대해 {@code addTrace(...)}가 두 번 호출되면 reader가
 * trace index row 하나와 point row 묶음 하나를 1:1로 맞출 수 없으므로 즉시 실패시킵니다.</p>
 *
 * <p>현재 구현은 {@link DuckDBAppender}로 DuckDB temp table에 row를 바로 append합니다.
 * close 시점에는 DuckDB가 temp table을 {@code ORDER BY sample_id, feature_id, point_idx}로
 * 정렬해서 parquet로 씁니다. Java heap에는 현재 trace 하나와 중복 검사용 feature set만
 * 남고, 대량 row buffering은 DuckDB temp storage가 담당합니다.</p>
 */
final class ArraySampleParquetRawSampleWriter implements AutoCloseable {
    private static final String POINT_TABLE = "tmp_array_sample_raw_points";
    private static final String TRACE_INDEX_TABLE = "tmp_array_sample_raw_trace_index";
    private static final int FILE_OPERATION_RETRY_COUNT = 10;
    private static final long FILE_OPERATION_RETRY_BASE_MILLIS = 25L;

    private final File outDir;
    private final File pointTmpPath;
    private final File traceIndexTmpPath;
    private final List<PointColumnSpec> pointSchema;
    private final ArraySampleParquetBuildOptions options;
    private final String compression;
    private final HashSet<String> seenTraceKeys = new HashSet<String>();

    private Connection rawConn;
    private Statement statement;
    private DuckDBAppender pointAppender;
    private DuckDBAppender traceIndexAppender;
    private boolean closed;
    private int traceCount;
    private int pointCount;

    ArraySampleParquetRawSampleWriter(
            File outDir,
            String pointTmpPath,
            String traceIndexTmpPath,
            List<PointColumnSpec> pointSchema,
            ArraySampleParquetBuildOptions options) throws Exception {
        this.outDir = outDir.getAbsoluteFile();
        this.pointTmpPath = new File(pointTmpPath).getAbsoluteFile();
        this.traceIndexTmpPath = new File(traceIndexTmpPath).getAbsoluteFile();
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        this.options = options;
        this.compression = options == null ? "zstd" : options.compression;
        ensureParent(this.pointTmpPath);
        ensureParent(this.traceIndexTmpPath);
        deleteIfExistsWithRetry(this.pointTmpPath);
        deleteIfExistsWithRetry(this.traceIndexTmpPath);
        openDuckDBWriters();
    }

    int traceCount() {
        return traceCount;
    }

    int pointCount() {
        return pointCount;
    }

    void writeTrace(long sampleId, int featureId, int traceLen, Map<String, Object> columns) throws Exception {
        ensureOpen();
        String traceKey = sampleId + ":" + featureId;
        if (!seenTraceKeys.add(traceKey)) {
            throw new IllegalArgumentException("duplicate trace for sample_id=" + sampleId + ", feature_id=" + featureId);
        }
        LinkedHashMap<String, Object> prepared = prepareColumns(traceLen, columns);

        traceIndexAppender.beginRow();
        traceIndexAppender.append(sampleId);
        traceIndexAppender.append(featureId);
        traceIndexAppender.append(traceLen);
        traceIndexAppender.endRow();

        for (int pointIdx = 0; pointIdx < traceLen; pointIdx++) {
            pointAppender.beginRow();
            pointAppender.append(sampleId);
            pointAppender.append(featureId);
            pointAppender.append(pointIdx);
            for (PointColumnSpec spec : pointSchema) {
                appendPointValue(pointAppender, spec, prepared.get(spec.name), pointIdx);
            }
            pointAppender.endRow();
        }

        traceCount++;
        long nextPointCount = (long) pointCount + (long) traceLen;
        if (nextPointCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("sample point count exceeds int range: " + nextPointCount);
        }
        pointCount = (int) nextPointCount;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        closed = true;
        try {
            closeAppenders();
            statement.execute(copyRawPointsSql(POINT_TABLE, pointTmpPath.getAbsolutePath(), pointSchema, compression));
            statement.execute(copyRawTraceIndexSql(TRACE_INDEX_TABLE, traceIndexTmpPath.getAbsolutePath(), compression));
        } catch (Exception e) {
            deleteQuietly(pointTmpPath);
            deleteQuietly(traceIndexTmpPath);
            throw e;
        } finally {
            closeDuckDBQuietly();
            seenTraceKeys.clear();
        }
    }

    void abort() {
        closed = true;
        closeDuckDBQuietly();
        seenTraceKeys.clear();
        deleteQuietly(pointTmpPath);
        deleteQuietly(traceIndexTmpPath);
    }

    private void openDuckDBWriters() throws Exception {
        rawConn = DuckDBUtils.connect(null);
        ArraySampleParquetDuckDB.configure(rawConn, outDir, options);
        statement = rawConn.createStatement();
        statement.execute(createRawPointsTableSql(pointSchema));
        statement.execute("CREATE TEMP TABLE " + TRACE_INDEX_TABLE + " (sample_id BIGINT, feature_id INTEGER, trace_len INTEGER)");
        DuckDBConnection conn = (DuckDBConnection) rawConn;
        pointAppender = conn.createAppender(DuckDBConnection.DEFAULT_SCHEMA, POINT_TABLE);
        traceIndexAppender = conn.createAppender(DuckDBConnection.DEFAULT_SCHEMA, TRACE_INDEX_TABLE);
    }

    private void closeAppenders() throws SQLException {
        SQLException first = null;
        if (pointAppender != null) {
            try {
                pointAppender.flush();
                pointAppender.close();
            } catch (SQLException e) {
                first = e;
            } finally {
                pointAppender = null;
            }
        }
        if (traceIndexAppender != null) {
            try {
                traceIndexAppender.flush();
                traceIndexAppender.close();
            } catch (SQLException e) {
                if (first == null) {
                    first = e;
                }
            } finally {
                traceIndexAppender = null;
            }
        }
        if (first != null) {
            throw first;
        }
    }

    private void closeDuckDBQuietly() {
        try {
            closeAppenders();
        } catch (Exception ignored) {
            // close/abort 중에는 원래 예외를 가리지 않도록 best-effort로만 정리합니다.
        }
        if (statement != null) {
            try {
                statement.close();
            } catch (Exception ignored) {
            }
            statement = null;
        }
        if (rawConn != null) {
            try {
                rawConn.close();
            } catch (Exception ignored) {
            }
            rawConn = null;
        }
    }

    private LinkedHashMap<String, Object> prepareColumns(int traceLen, Map<String, Object> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("columns must not be null");
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        for (PointColumnSpec spec : pointSchema) {
            if (!columns.containsKey(spec.name)) {
                throw new IllegalArgumentException("missing point column: " + spec.name);
            }
            Object value = columns.get(spec.name);
            Object normalized;
            switch (spec.storageType) {
                case FLOAT64:
                    normalized = ArrayUtils.toDoubleArray(value, spec.name);
                    break;
                case INT32:
                    normalized = ArrayUtils.toIntArray(value, spec.name);
                    break;
                case STRING:
                    normalized = ArrayUtils.toStringArray(value, spec.name);
                    validateNoNullStrings((String[]) normalized, spec.name);
                    break;
                case INT64:
                    normalized = ArrayUtils.toLongArray(value, spec.name);
                    break;
                case UINT8:
                case UINT16:
                case UINT32:
                case UINT64:
                    normalized = ArrayUtils.toLongArray(value, spec.name);
                    validateUnsignedValues((long[]) normalized, spec.storageType, spec.name);
                    break;
                default:
                    throw new IllegalArgumentException("unsupported storage type: " + spec.storageType.value);
            }
            int length = ArrayUtils.pointColumnLength(normalized);
            if (length != traceLen) {
                throw new IllegalArgumentException("point column length mismatch for " + spec.name + ": " + length + " != " + traceLen);
            }
            out.put(spec.name, normalized);
        }
        if (columns.size() != pointSchema.size()) {
            for (String name : columns.keySet()) {
                if (!out.containsKey(name)) {
                    throw new IllegalArgumentException("unexpected point column: " + name);
                }
            }
        }
        return out;
    }

    private static void appendPointValue(DuckDBAppender appender, PointColumnSpec spec, Object values, int index) throws SQLException {
        switch (spec.storageType) {
            case FLOAT64:
                appender.append(((double[]) values)[index]);
                return;
            case INT32:
                appender.append(((int[]) values)[index]);
                return;
            case INT64:
                appender.append(((long[]) values)[index]);
                return;
            case STRING:
                appender.append(((String[]) values)[index]);
                return;
            case UINT8:
            case UINT16:
            case UINT32:
            case UINT64:
                appender.append(((long[]) values)[index]);
                return;
            default:
                throw new IllegalArgumentException("unsupported storage type: " + spec.storageType.value);
        }
    }

    private static String createRawPointsTableSql(List<PointColumnSpec> pointSchema) {
        StringBuilder sb = new StringBuilder();
        sb.append("CREATE TEMP TABLE ").append(POINT_TABLE).append(" (");
        sb.append("sample_id BIGINT, feature_id INTEGER, point_idx INTEGER");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ")
                    .append(DuckDBUtils.quoteIdentifier(spec.name))
                    .append(" ")
                    .append(sqlType(spec.storageType));
        }
        sb.append(")");
        return sb.toString();
    }

    private static String copyRawPointsSql(String tableName, String path, List<PointColumnSpec> pointSchema, String compression) {
        StringBuilder sb = new StringBuilder();
        sb.append("COPY (SELECT sample_id, feature_id, point_idx");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ").append(DuckDBUtils.quoteIdentifier(spec.name));
        }
        sb.append(" FROM ").append(tableName);
        sb.append(" ORDER BY sample_id, feature_id, point_idx)");
        sb.append(" TO ").append(DuckDBUtils.quotePath(path)).append(" ");
        sb.append(ArraySampleParquetDuckDB.parquetCopyOptions(compression));
        return sb.toString();
    }

    private static String copyRawTraceIndexSql(String tableName, String path, String compression) {
        return "COPY (SELECT sample_id, feature_id, trace_len FROM " + tableName
                + " ORDER BY sample_id, feature_id)"
                + " TO " + DuckDBUtils.quotePath(path) + " "
                + ArraySampleParquetDuckDB.parquetCopyOptions(compression);
    }

    private static String sqlType(StorageType storageType) {
        switch (storageType) {
            case FLOAT64:
                return "DOUBLE";
            case INT32:
                return "INTEGER";
            case INT64:
                return "BIGINT";
            case STRING:
                return "VARCHAR";
            case UINT8:
                return "UTINYINT";
            case UINT16:
                return "USMALLINT";
            case UINT32:
                return "UINTEGER";
            case UINT64:
                return "UBIGINT";
            default:
                throw new IllegalArgumentException("unsupported storage type: " + storageType.value);
        }
    }

    private static void validateUnsignedValues(long[] values, StorageType storageType, String columnName) {
        long max;
        switch (storageType) {
            case UINT8:
                max = 0xFFL;
                break;
            case UINT16:
                max = 0xFFFFL;
                break;
            case UINT32:
                max = 0xFFFFFFFFL;
                break;
            case UINT64:
                return;
            default:
                return;
        }
        for (long value : values) {
            if (value < 0L || value > max) {
                throw new IllegalArgumentException("point column " + columnName + " value out of " + storageType.value + " range: " + value);
            }
        }
    }

    private static void validateNoNullStrings(String[] values, String columnName) {
        for (String value : values) {
            if (value == null) {
                throw new IllegalArgumentException("point column " + columnName + " must not contain null string values");
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("raw sample writer is already closed");
        }
    }

    private static void ensureParent(File path) throws SQLException {
        File parent = path.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new SQLException("failed to create directory: " + parent.getAbsolutePath());
        }
    }

    private static void deleteIfExistsWithRetry(File file) throws SQLException {
        if (!file.exists()) {
            return;
        }
        SQLException last = null;
        for (int attempt = 0; attempt < FILE_OPERATION_RETRY_COUNT; attempt++) {
            if (!file.exists() || file.delete()) {
                return;
            }
            last = new SQLException("failed to remove stale tmp file: " + file.getAbsolutePath());
            if (attempt >= FILE_OPERATION_RETRY_COUNT - 1) {
                break;
            }
            sleepBeforeRetry(file, attempt);
        }
        throw last == null ? new SQLException("failed to remove stale tmp file: " + file.getAbsolutePath()) : last;
    }

    private static void deleteQuietly(File path) {
        if (path.exists() && !path.delete()) {
            // best-effort cleanup
        }
    }

    private static void sleepBeforeRetry(File file, int attempt) throws SQLException {
        try {
            Thread.sleep(FILE_OPERATION_RETRY_BASE_MILLIS * (long) (attempt + 1));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new SQLException("interrupted while waiting for file operation retry: " + file.getAbsolutePath(), e);
        }
    }
}
