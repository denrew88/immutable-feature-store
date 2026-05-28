package fs.io.array_sample_parquet;

import fs.io.common.ArrayUtils;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;
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
import java.util.List;
import java.util.Map;

/**
 * sample-major trace row를 Parquet part 파일 하나로 직접 기록한다.
 *
 * <p>행 하나는 `(sample_id, feature_id)` trace 하나이고, point column은 Parquet
 * LIST column으로 쓴다. empty trace는 row는 존재하지만 모든 list가 비어 있는
 * 상태로 표현된다.
 */
final class ArraySampleParquetPartWriter implements AutoCloseable {
    private final File finalFile;
    private final File tmpFile;
    private final List<PointColumnSpec> pointSchema;
    private final ParquetWriter<Group> writer;
    private final SimpleGroupFactory groupFactory;
    private boolean closed;

    private ArraySampleParquetPartWriter(String finalPath, List<PointColumnSpec> pointSchema, String compression) throws IOException {
        this.finalFile = new File(finalPath).getAbsoluteFile();
        this.tmpFile = new File(finalPath + ".tmp").getAbsoluteFile();
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        File parent = finalFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create part dir: " + parent.getAbsolutePath());
        }
        MessageType schema = buildSchema(this.pointSchema);
        this.groupFactory = new SimpleGroupFactory(schema);
        this.writer = ExampleParquetWriter.builder(new LocalOutputFile(tmpFile))
                .withConf(new Configuration(false))
                .withType(schema)
                .withCompressionCodec(compressionCodec(compression))
                .build();
    }

    static ArraySampleParquetPartWriter open(String finalPath, List<PointColumnSpec> pointSchema, String compression) throws IOException {
        return new ArraySampleParquetPartWriter(finalPath, pointSchema, compression);
    }

    void writeRow(long sampleId, int featureId, int traceLen, Map<String, Object> columns) throws IOException {
        Group row = groupFactory.newGroup()
                .append("sample_id", sampleId)
                .append("feature_id", featureId)
                .append("trace_len", traceLen);
        for (PointColumnSpec spec : pointSchema) {
            Object values = columns.get(spec.name);
            int length = ArrayUtils.pointColumnLength(values);
            if (length != traceLen) {
                throw new IllegalArgumentException("point column length mismatch for " + spec.name);
            }
            appendList(row, spec, values);
        }
        writer.write(row);
    }

    String finalPath() {
        return finalFile.getAbsolutePath();
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
        if (closeEx == null) {
            moveTmpToFinal();
        } else {
            deleteTmpQuietly();
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
            closed = true;
        }
        deleteTmpQuietly();
    }

    private static MessageType buildSchema(List<PointColumnSpec> pointSchema) {
        StringBuilder sb = new StringBuilder();
        sb.append("message array_sample_parquet {\n");
        sb.append("  required int64 sample_id;\n");
        sb.append("  required int32 feature_id;\n");
        sb.append("  required int32 trace_len;\n");
        for (PointColumnSpec spec : pointSchema) {
            sb.append("  optional group ").append(spec.name).append(" (LIST) {\n");
            sb.append("    repeated group list {\n");
            sb.append("      ").append(elementLine(spec)).append(";\n");
            sb.append("    }\n");
            sb.append("  }\n");
        }
        sb.append("}\n");
        return MessageTypeParser.parseMessageType(sb.toString());
    }

    private static String elementLine(PointColumnSpec spec) {
        switch (spec.storageType) {
            case FLOAT64:
                return "required double element";
            case INT32:
                return "required int32 element";
            case INT64:
                return "required int64 element";
            case UINT32:
                return "required int32 element (UINT_32)";
            case UINT64:
                return "required int64 element (UINT_64)";
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

    private static void appendList(Group row, PointColumnSpec spec, Object values) {
        Group list = row.addGroup(spec.name);
        switch (spec.storageType) {
            case FLOAT64: {
                double[] source = ArrayUtils.toDoubleArray(values, spec.name);
                for (double value : source) {
                    list.addGroup("list").append("element", value);
                }
                return;
            }
            case INT32: {
                int[] source = ArrayUtils.toIntArray(values, spec.name);
                for (int value : source) {
                    list.addGroup("list").append("element", value);
                }
                return;
            }
            case INT64: {
                long[] source = ArrayUtils.toLongArray(values, spec.name);
                for (long value : source) {
                    list.addGroup("list").append("element", value);
                }
                return;
            }
            case UINT32: {
                long[] source = ArrayUtils.toLongArray(values, spec.name);
                for (long value : source) {
                    list.addGroup("list").append("element", (int) (value & 0xFFFFFFFFL));
                }
                return;
            }
            case UINT64: {
                long[] source = ArrayUtils.toLongArray(values, spec.name);
                for (long value : source) {
                    list.addGroup("list").append("element", value);
                }
                return;
            }
            default:
                throw new IllegalArgumentException("unsupported storage_type: " + spec.storageType.value);
        }
    }

    private void moveTmpToFinal() throws IOException {
        try {
            Files.move(tmpFile.toPath(), finalFile.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmpFile.toPath(), finalFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private void deleteTmpQuietly() {
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
            this.pos = 0L;
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
