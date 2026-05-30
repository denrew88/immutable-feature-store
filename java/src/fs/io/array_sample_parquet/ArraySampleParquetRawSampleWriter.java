package fs.io.array_sample_parquet;

import fs.io.common.ArrayUtils;
import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.duckdb.DuckDBConnection;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * sample 하나를 raw point parquet와 raw trace-index parquet 파일로 저장하는 writer.
 *
 * <p>이 writer는 point row를 JDBC row appender로 하나씩 넣지 않는다. {@code addTrace} 시점에는
 * trace별 primitive array를 보관하고, sample close 시점에 Arrow record batch stream으로 DuckDB에
 * 등록한 뒤 {@code COPY ... TO parquet}를 실행한다. 즉 Java/DuckDB 경계는 row 단위 호출이 아니라
 * column vector batch 단위로 넘어간다.</p>
 */
final class ArraySampleParquetRawSampleWriter implements AutoCloseable {
    private static final String POINT_STREAM_TABLE = "arrow_array_sample_raw_points";
    private static final String TRACE_INDEX_STREAM_TABLE = "arrow_array_sample_raw_trace_index";
    private static final int DEFAULT_ARROW_BATCH_ROWS = 262144;

    private final File outDir;
    private final File pointTmpPath;
    private final File traceIndexTmpPath;
    private final List<PointColumnSpec> pointSchema;
    private final ArraySampleParquetBuildOptions options;
    private final String compression;
    private final int arrowBatchRows;
    private final ArrayList<TraceData> traces = new ArrayList<TraceData>();
    private final ArrayList<TraceIndexRow> traceIndexRows = new ArrayList<TraceIndexRow>();
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
        this.arrowBatchRows = options == null || options.arrowBatchRows <= 0
                ? DEFAULT_ARROW_BATCH_ROWS
                : options.arrowBatchRows;
        ensureParent(this.pointTmpPath);
        ensureParent(this.traceIndexTmpPath);
    }

    int traceCount() {
        return traceCount;
    }

    int pointCount() {
        return pointCount;
    }

    void writeTrace(long sampleId, int featureId, int traceLen, Map<String, Object> columns) throws SQLException {
        ensureOpen();
        LinkedHashMap<String, Object> prepared = prepareColumns(traceLen, columns);
        traces.add(new TraceData(sampleId, featureId, traceLen, prepared));
        traceIndexRows.add(new TraceIndexRow(sampleId, featureId, traceLen));
        traceCount++;
        long nextPointCount = (long) pointCount + traceLen;
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
        deleteQuietly(pointTmpPath);
        deleteQuietly(traceIndexTmpPath);
        try {
            writeParquetFromArrow();
        } catch (Exception e) {
            deleteQuietly(pointTmpPath);
            deleteQuietly(traceIndexTmpPath);
            throw e;
        } finally {
            traces.clear();
            traceIndexRows.clear();
        }
    }

    void abort() {
        if (!closed) {
            closed = true;
        }
        traces.clear();
        traceIndexRows.clear();
        deleteQuietly(pointTmpPath);
        deleteQuietly(traceIndexTmpPath);
    }

    private void writeParquetFromArrow() throws Exception {
        sortTraceRowsForStreaming();
        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
             PointArrowReader pointReader = new PointArrowReader(allocator, traces, pointSchema, arrowBatchRows);
             TraceIndexArrowReader traceIndexReader = new TraceIndexArrowReader(allocator, traceIndexRows, arrowBatchRows);
             ArrowArrayStream pointStream = ArrowArrayStream.allocateNew(allocator);
             ArrowArrayStream traceIndexStream = ArrowArrayStream.allocateNew(allocator);
             Connection conn = DuckDBUtils.connect(null);
             Statement st = conn.createStatement()) {
            ArraySampleParquetDuckDB.configure(conn, outDir, options);
            Data.exportArrayStream(allocator, pointReader, pointStream);
            ((DuckDBConnection) conn).registerArrowStream(POINT_STREAM_TABLE, pointStream);
            st.execute(copyRawPointsSql(POINT_STREAM_TABLE, pointTmpPath.getAbsolutePath(), pointSchema, compression));

            Data.exportArrayStream(allocator, traceIndexReader, traceIndexStream);
            ((DuckDBConnection) conn).registerArrowStream(TRACE_INDEX_STREAM_TABLE, traceIndexStream);
            st.execute(copyRawTraceIndexSql(TRACE_INDEX_STREAM_TABLE, traceIndexTmpPath.getAbsolutePath(), compression));
        }
    }

    private void sortTraceRowsForStreaming() {
        // 사용자가 addTrace를 feature_id 순서대로 호출한다는 보장은 없다.
        // DuckDB SQL sort를 피하려면 Arrow stream 자체가 포맷의 물리 정렬 순서를 만족해야 한다.
        Collections.sort(traces, new Comparator<TraceData>() {
            @Override
            public int compare(TraceData a, TraceData b) {
                int sampleCmp = Long.compare(a.sampleId, b.sampleId);
                if (sampleCmp != 0) {
                    return sampleCmp;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
        Collections.sort(traceIndexRows, new Comparator<TraceIndexRow>() {
            @Override
            public int compare(TraceIndexRow a, TraceIndexRow b) {
                int sampleCmp = Long.compare(a.sampleId, b.sampleId);
                if (sampleCmp != 0) {
                    return sampleCmp;
                }
                return Integer.compare(a.featureId, b.featureId);
            }
        });
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
            out.put(spec.name, copyPointColumn(normalized));
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

    private static String copyRawPointsSql(String tableName, String path, List<PointColumnSpec> pointSchema, String compression) {
        StringBuilder sb = new StringBuilder();
        sb.append("COPY (SELECT sample_id, feature_id, point_idx");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ").append(DuckDBUtils.quoteIdentifier(spec.name));
        }
        sb.append(" FROM ").append(tableName).append(")");
        sb.append(" TO ").append(DuckDBUtils.quotePath(path)).append(" ");
        sb.append(ArraySampleParquetDuckDB.parquetCopyOptions(compression));
        return sb.toString();
    }

    private static String copyRawTraceIndexSql(String tableName, String path, String compression) {
        return "COPY (SELECT sample_id, feature_id, trace_len FROM " + tableName
                + ") TO " + DuckDBUtils.quotePath(path) + " "
                + ArraySampleParquetDuckDB.parquetCopyOptions(compression);
    }

    private static Schema pointArrowSchema(List<PointColumnSpec> pointSchema) {
        ArrayList<Field> fields = new ArrayList<Field>();
        fields.add(new Field("sample_id", FieldType.notNullable(new ArrowType.Int(64, true)), null));
        fields.add(new Field("feature_id", FieldType.notNullable(new ArrowType.Int(32, true)), null));
        fields.add(new Field("point_idx", FieldType.notNullable(new ArrowType.Int(32, true)), null));
        for (PointColumnSpec spec : pointSchema) {
            fields.add(new Field(spec.name, FieldType.notNullable(arrowType(spec.storageType)), null));
        }
        return new Schema(fields);
    }

    private static Schema traceIndexArrowSchema() {
        return new Schema(Arrays.asList(
                new Field("sample_id", FieldType.notNullable(new ArrowType.Int(64, true)), null),
                new Field("feature_id", FieldType.notNullable(new ArrowType.Int(32, true)), null),
                new Field("trace_len", FieldType.notNullable(new ArrowType.Int(32, true)), null)));
    }

    private static ArrowType arrowType(StorageType storageType) {
        switch (storageType) {
            case FLOAT64:
                return new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
            case INT32:
                return new ArrowType.Int(32, true);
            case INT64:
                return new ArrowType.Int(64, true);
            case STRING:
                return new ArrowType.Utf8();
            case UINT8:
                return new ArrowType.Int(8, false);
            case UINT16:
                return new ArrowType.Int(16, false);
            case UINT32:
                return new ArrowType.Int(32, false);
            case UINT64:
                return new ArrowType.Int(64, false);
            default:
                throw new IllegalArgumentException("unsupported storage type: " + storageType.value);
        }
    }

    private static Object copyPointColumn(Object values) {
        if (values instanceof double[]) {
            double[] source = (double[]) values;
            return Arrays.copyOf(source, source.length);
        }
        if (values instanceof int[]) {
            int[] source = (int[]) values;
            return Arrays.copyOf(source, source.length);
        }
        if (values instanceof long[]) {
            long[] source = (long[]) values;
            return Arrays.copyOf(source, source.length);
        }
        if (values instanceof String[]) {
            String[] source = (String[]) values;
            return Arrays.copyOf(source, source.length);
        }
        throw new IllegalArgumentException("unsupported point column array type: " + values.getClass().getName());
    }

    private static void setPointColumn(
            FieldVector vector,
            PointColumnSpec spec,
            Object values,
            int sourceIndex,
            int targetIndex,
            Map<String, byte[]> stringBytesCache) {
        switch (spec.storageType) {
            case FLOAT64:
                ((Float8Vector) vector).set(targetIndex, ((double[]) values)[sourceIndex]);
                return;
            case INT32:
                ((IntVector) vector).set(targetIndex, ((int[]) values)[sourceIndex]);
                return;
            case INT64:
                ((BigIntVector) vector).set(targetIndex, ((long[]) values)[sourceIndex]);
                return;
            case STRING: {
                String value = ((String[]) values)[sourceIndex];
                byte[] bytes = stringBytesCache.get(value);
                if (bytes == null) {
                    bytes = value.getBytes(StandardCharsets.UTF_8);
                    stringBytesCache.put(value, bytes);
                }
                ((VarCharVector) vector).setSafe(targetIndex, bytes, 0, bytes.length);
                return;
            }
            case UINT8:
                ((UInt1Vector) vector).set(targetIndex, (int) ((long[]) values)[sourceIndex]);
                return;
            case UINT16:
                ((UInt2Vector) vector).set(targetIndex, (int) ((long[]) values)[sourceIndex]);
                return;
            case UINT32:
                ((UInt4Vector) vector).set(targetIndex, (int) ((long[]) values)[sourceIndex]);
                return;
            case UINT64:
                ((UInt8Vector) vector).set(targetIndex, ((long[]) values)[sourceIndex]);
                return;
            default:
                throw new IllegalArgumentException("unsupported storage type: " + spec.storageType.value);
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

    private static void deleteQuietly(File path) {
        if (path.exists() && !path.delete()) {
            // best-effort cleanup
        }
    }

    private static void prepareRoot(VectorSchemaRoot root, int batchRows) {
        for (FieldVector vector : root.getFieldVectors()) {
            vector.setInitialCapacity(batchRows);
        }
        root.allocateNew();
    }

    private static final class TraceData {
        final long sampleId;
        final int featureId;
        final int traceLen;
        final LinkedHashMap<String, Object> columns;

        TraceData(long sampleId, int featureId, int traceLen, LinkedHashMap<String, Object> columns) {
            this.sampleId = sampleId;
            this.featureId = featureId;
            this.traceLen = traceLen;
            this.columns = columns;
        }
    }

    private static final class TraceIndexRow {
        final long sampleId;
        final int featureId;
        final int traceLen;

        TraceIndexRow(long sampleId, int featureId, int traceLen) {
            this.sampleId = sampleId;
            this.featureId = featureId;
            this.traceLen = traceLen;
        }
    }

    private static final class PointArrowReader extends ArrowReader {
        private final List<TraceData> traces;
        private final List<PointColumnSpec> pointSchema;
        private final Schema schema;
        private final int batchRows;
        private final HashMap<String, byte[]> stringBytesCache = new HashMap<String, byte[]>();
        private int traceCursor;
        private int pointCursor;

        PointArrowReader(BufferAllocator allocator, List<TraceData> traces, List<PointColumnSpec> pointSchema, int batchRows) {
            super(allocator);
            this.traces = traces;
            this.pointSchema = pointSchema;
            this.schema = pointArrowSchema(pointSchema);
            this.batchRows = Math.max(1, batchRows);
        }

        @Override
        public boolean loadNextBatch() throws IOException {
            prepareLoadNextBatch();
            stringBytesCache.clear();

            try (VectorSchemaRoot batchRoot = VectorSchemaRoot.create(schema, allocator)) {
                prepareRoot(batchRoot, batchRows);
                BigIntVector sampleIdVector = (BigIntVector) batchRoot.getVector("sample_id");
                IntVector featureIdVector = (IntVector) batchRoot.getVector("feature_id");
                IntVector pointIdxVector = (IntVector) batchRoot.getVector("point_idx");
                FieldVector[] pointVectors = new FieldVector[pointSchema.size()];
                for (int i = 0; i < pointSchema.size(); i++) {
                    pointVectors[i] = batchRoot.getVector(pointSchema.get(i).name);
                }

                int row = 0;
                while (row < batchRows && traceCursor < traces.size()) {
                    TraceData trace = traces.get(traceCursor);
                    if (pointCursor >= trace.traceLen) {
                        traceCursor++;
                        pointCursor = 0;
                        continue;
                    }
                    sampleIdVector.set(row, trace.sampleId);
                    featureIdVector.set(row, trace.featureId);
                    pointIdxVector.set(row, pointCursor);
                    for (int col = 0; col < pointSchema.size(); col++) {
                        PointColumnSpec spec = pointSchema.get(col);
                        setPointColumn(pointVectors[col], spec, trace.columns.get(spec.name), pointCursor, row, stringBytesCache);
                    }
                    row++;
                    pointCursor++;
                }

                if (row == 0) {
                    return false;
                }
                batchRoot.setRowCount(row);
                loadRecordBatch(new VectorUnloader(batchRoot).getRecordBatch());
                return true;
            }
        }

        @Override
        public long bytesRead() {
            return 0L;
        }

        @Override
        protected void closeReadSource() {
        }

        @Override
        protected Schema readSchema() {
            return schema;
        }
    }

    private static final class TraceIndexArrowReader extends ArrowReader {
        private final List<TraceIndexRow> rows;
        private final Schema schema = traceIndexArrowSchema();
        private final int batchRows;
        private int cursor;

        TraceIndexArrowReader(BufferAllocator allocator, List<TraceIndexRow> rows, int batchRows) {
            super(allocator);
            this.rows = rows;
            this.batchRows = Math.max(1, batchRows);
        }

        @Override
        public boolean loadNextBatch() throws IOException {
            prepareLoadNextBatch();
            try (VectorSchemaRoot batchRoot = VectorSchemaRoot.create(schema, allocator)) {
                prepareRoot(batchRoot, batchRows);
                BigIntVector sampleIdVector = (BigIntVector) batchRoot.getVector("sample_id");
                IntVector featureIdVector = (IntVector) batchRoot.getVector("feature_id");
                IntVector traceLenVector = (IntVector) batchRoot.getVector("trace_len");

                int row = 0;
                while (row < batchRows && cursor < rows.size()) {
                    TraceIndexRow source = rows.get(cursor);
                    sampleIdVector.set(row, source.sampleId);
                    featureIdVector.set(row, source.featureId);
                    traceLenVector.set(row, source.traceLen);
                    row++;
                    cursor++;
                }
                if (row == 0) {
                    return false;
                }
                batchRoot.setRowCount(row);
                loadRecordBatch(new VectorUnloader(batchRoot).getRecordBatch());
                return true;
            }
        }

        @Override
        public long bytesRead() {
            return 0L;
        }

        @Override
        protected void closeReadSource() {
        }

        @Override
        protected Schema readSchema() {
            return schema;
        }
    }
}
