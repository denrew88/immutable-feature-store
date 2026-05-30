package fs.io.array_sample_parquet;

import fs.io.common.ArrayUtils;
import fs.model.common.PointColumnSpec;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.OutputFile;
import org.apache.parquet.io.PositionOutputStream;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array_sample_parquet part를 long format으로 쓴다.
 *
 * <p>point part의 row 하나는 point 하나다. empty trace는 point row가 없으므로
 * 별도 trace index part에 `(sample_id, feature_id, trace_len)`을 기록해서
 * missing trace와 구분한다.</p>
 */
final class ArraySampleParquetPartWriter implements AutoCloseable {
    private final File finalFile;
    private final File tmpFile;
    private final File traceIndexFinalFile;
    private final File traceIndexTmpFile;
    private final List<PointColumnSpec> pointSchema;
    private final ParquetWriter<Group> writer;
    private final ParquetWriter<Group> traceIndexWriter;
    private final SimpleGroupFactory groupFactory;
    private final SimpleGroupFactory traceIndexGroupFactory;
    private boolean closed;

    private ArraySampleParquetPartWriter(String finalPath, String traceIndexPath, List<PointColumnSpec> pointSchema, String compression) throws IOException {
        this.finalFile = new File(finalPath).getAbsoluteFile();
        this.tmpFile = new File(finalPath + ".tmp").getAbsoluteFile();
        this.traceIndexFinalFile = new File(traceIndexPath).getAbsoluteFile();
        this.traceIndexTmpFile = new File(traceIndexPath + ".tmp").getAbsoluteFile();
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        ensureParent(finalFile);
        ensureParent(traceIndexFinalFile);

        MessageType pointMessage = buildPointSchema(this.pointSchema);
        MessageType traceIndexMessage = buildTraceIndexSchema();
        this.groupFactory = new SimpleGroupFactory(pointMessage);
        this.traceIndexGroupFactory = new SimpleGroupFactory(traceIndexMessage);
        CompressionCodecName codec = compressionCodec(compression);
        this.writer = ExampleParquetWriter.builder(new LocalOutputFile(tmpFile))
                .withConf(new Configuration(false))
                .withType(pointMessage)
                .withCompressionCodec(codec)
                .build();
        this.traceIndexWriter = ExampleParquetWriter.builder(new LocalOutputFile(traceIndexTmpFile))
                .withConf(new Configuration(false))
                .withType(traceIndexMessage)
                .withCompressionCodec(codec)
                .build();
    }

    static ArraySampleParquetPartWriter open(String finalPath, String traceIndexPath, List<PointColumnSpec> pointSchema, String compression) throws IOException {
        return new ArraySampleParquetPartWriter(finalPath, traceIndexPath, pointSchema, compression);
    }

    void writeRow(long sampleId, int featureId, int traceLen, Map<String, Object> columns) throws IOException {
        Group indexRow = traceIndexGroupFactory.newGroup()
                .append("sample_id", sampleId)
                .append("feature_id", featureId)
                .append("trace_len", traceLen);
        traceIndexWriter.write(indexRow);

        LinkedHashMap<String, Object> prepared = new LinkedHashMap<String, Object>();
        for (PointColumnSpec spec : pointSchema) {
            Object values = columns.get(spec.name);
            int length = ArrayUtils.pointColumnLength(values);
            if (length != traceLen) {
                throw new IllegalArgumentException("point column length mismatch for " + spec.name);
            }
            prepared.put(spec.name, values);
        }

        for (int pointIdx = 0; pointIdx < traceLen; pointIdx++) {
            Group row = groupFactory.newGroup()
                    .append("sample_id", sampleId)
                    .append("feature_id", featureId)
                    .append("point_idx", pointIdx);
            for (PointColumnSpec spec : pointSchema) {
                appendScalar(row, spec, prepared.get(spec.name), pointIdx);
            }
            writer.write(row);
        }
    }

    String finalPath() {
        return finalFile.getAbsolutePath();
    }

    String traceIndexFinalPath() {
        return traceIndexFinalFile.getAbsolutePath();
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        IOException closeEx = null;
        try {
            writer.close();
        } catch (IOException e) {
            closeEx = e;
        }
        try {
            traceIndexWriter.close();
        } catch (IOException e) {
            if (closeEx == null) {
                closeEx = e;
            }
        }
        if (closeEx == null) {
            moveTmpToFinal(tmpFile, finalFile);
            moveTmpToFinal(traceIndexTmpFile, traceIndexFinalFile);
        } else {
            deleteTmpQuietly(tmpFile);
            deleteTmpQuietly(traceIndexTmpFile);
            throw closeEx;
        }
    }

    void abort() {
        if (!closed) {
            try {
                writer.close();
            } catch (IOException ignored) {
                // best-effort abort
            }
            try {
                traceIndexWriter.close();
            } catch (IOException ignored) {
                // best-effort abort
            }
            closed = true;
        }
        deleteTmpQuietly(tmpFile);
        deleteTmpQuietly(traceIndexTmpFile);
    }

    private static MessageType buildPointSchema(List<PointColumnSpec> pointSchema) {
        StringBuilder sb = new StringBuilder();
        sb.append("message array_sample_parquet {\n");
        sb.append("  required int64 sample_id;\n");
        sb.append("  required int32 feature_id;\n");
        sb.append("  required int32 point_idx;\n");
        for (PointColumnSpec spec : pointSchema) {
            sb.append("  ").append(fieldLine(spec)).append(";\n");
        }
        sb.append("}\n");
        return MessageTypeParser.parseMessageType(sb.toString());
    }

    private static MessageType buildTraceIndexSchema() {
        return MessageTypeParser.parseMessageType(
                "message array_sample_parquet_trace_index {\n" +
                        "  required int64 sample_id;\n" +
                        "  required int32 feature_id;\n" +
                        "  required int32 trace_len;\n" +
                        "}\n");
    }

    private static String fieldLine(PointColumnSpec spec) {
        switch (spec.storageType) {
            case FLOAT64:
                return "required double " + spec.name;
            case INT32:
                return "required int32 " + spec.name;
            case INT64:
                return "required int64 " + spec.name;
            case STRING:
                return "required binary " + spec.name + " (UTF8)";
            case UINT8:
                return "required int32 " + spec.name + " (UINT_8)";
            case UINT16:
                return "required int32 " + spec.name + " (UINT_16)";
            case UINT32:
                return "required int32 " + spec.name + " (UINT_32)";
            case UINT64:
                return "required int64 " + spec.name + " (UINT_64)";
            default:
                throw new IllegalArgumentException("unsupported storage_type: " + spec.storageType.value);
        }
    }

    private static CompressionCodecName compressionCodec(String compression) {
        String value = (compression == null) ? "" : compression.trim().toLowerCase();
        if (value.isEmpty() || "none".equals(value) || "uncompressed".equals(value)) {
            return CompressionCodecName.UNCOMPRESSED;
        }
        if ("snappy".equals(value)) {
            return CompressionCodecName.SNAPPY;
        }
        if ("gzip".equals(value)) {
            return CompressionCodecName.GZIP;
        }
        if ("zstd".equals(value)) {
            return CompressionCodecName.ZSTD;
        }
        throw new IllegalArgumentException("unsupported parquet compression: " + compression);
    }

    private static void appendScalar(Group row, PointColumnSpec spec, Object values, int index) {
        switch (spec.storageType) {
            case FLOAT64:
                row.append(spec.name, ArrayUtils.toDoubleArray(values, spec.name)[index]);
                return;
            case INT32:
                row.append(spec.name, ArrayUtils.toIntArray(values, spec.name)[index]);
                return;
            case INT64:
                row.append(spec.name, ArrayUtils.toLongArray(values, spec.name)[index]);
                return;
            case STRING:
                row.append(spec.name, ArrayUtils.toStringArray(values, spec.name)[index]);
                return;
            case UINT8:
                row.append(spec.name, (int) (ArrayUtils.toLongArray(values, spec.name)[index] & 0xFFL));
                return;
            case UINT16:
                row.append(spec.name, (int) (ArrayUtils.toLongArray(values, spec.name)[index] & 0xFFFFL));
                return;
            case UINT32:
                row.append(spec.name, (int) (ArrayUtils.toLongArray(values, spec.name)[index] & 0xFFFFFFFFL));
                return;
            case UINT64:
                row.append(spec.name, ArrayUtils.toLongArray(values, spec.name)[index]);
                return;
            default:
                throw new IllegalArgumentException("unsupported storage_type: " + spec.storageType.value);
        }
    }

    private static void ensureParent(File file) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create part dir: " + parent.getAbsolutePath());
        }
    }

    private static void moveTmpToFinal(File tmpFile, File finalFile) throws IOException {
        try {
            Files.move(tmpFile.toPath(), finalFile.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmpFile.toPath(), finalFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void deleteTmpQuietly(File tmpFile) {
        if (tmpFile.exists() && !tmpFile.delete()) {
            // best-effort cleanup
        }
    }

    private static final class LocalOutputFile implements OutputFile {
        private final File file;

        LocalOutputFile(File file) {
            this.file = file.getAbsoluteFile();
        }

        @Override
        public PositionOutputStream create(long blockSizeHint) throws IOException {
            return createOrOverwrite(blockSizeHint);
        }

        @Override
        public PositionOutputStream createOrOverwrite(long blockSizeHint) throws IOException {
            return new LocalPositionOutputStream(new BufferedOutputStream(new FileOutputStream(file, false), 64 * 1024));
        }

        @Override
        public boolean supportsBlockSize() {
            return false;
        }

        @Override
        public long defaultBlockSize() {
            return 0L;
        }

        @Override
        public String getPath() {
            return file.getAbsolutePath();
        }
    }

    private static final class LocalPositionOutputStream extends PositionOutputStream {
        private final OutputStream out;
        private long pos;

        LocalPositionOutputStream(OutputStream out) {
            this.out = out;
        }

        @Override
        public long getPos() {
            return pos;
        }

        @Override
        public void write(int b) throws IOException {
            out.write(b);
            pos++;
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
            pos += len;
        }

        @Override
        public void flush() throws IOException {
            out.flush();
        }

        @Override
        public void close() throws IOException {
            out.close();
        }
    }
}
