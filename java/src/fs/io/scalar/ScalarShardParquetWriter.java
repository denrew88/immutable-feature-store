package fs.io.scalar;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.OutputFile;
import org.apache.parquet.io.PositionOutputStream;
import org.apache.parquet.io.api.Binary;
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

/**
 * scalar shard parquet 한 파일을 Java에서 직접 쓰는 전용 writer다.
 *
 * <p>기존에는 DuckDB temp table에 row를 넣은 뒤 {@code COPY ... TO PARQUET}로
 * 최종 shard를 만들었지만, 이 writer는 intermediate temp table 없이 바로
 * {@code (feature_id, value_len, values_blob, valid_blob)} row를 parquet에 기록한다.
 *
 * <p>출력은 항상 같은 디렉터리의 {@code .tmp} 파일에 먼저 쓰고,
 * close 시점에 최종 parquet 경로로 rename 한다. 따라서 중간 실패 시
 * 부분적으로 깨진 최종 shard 파일이 남지 않게 한다.
 */
final class ScalarShardParquetWriter implements AutoCloseable {
    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message scalar_shard {\n"
                    + "  required int32 feature_id;\n"
                    + "  required int32 value_len;\n"
                    + "  required binary values_blob;\n"
                    + "  required binary valid_blob;\n"
                    + "}"
    );

    private final File tmpFile;
    private final File finalFile;
    private final ParquetWriter<Group> writer;
    private final SimpleGroupFactory groupFactory;
    private boolean closed;

    private ScalarShardParquetWriter(String finalPath) throws IOException {
        this.finalFile = new File(finalPath).getAbsoluteFile();
        this.tmpFile = new File(finalPath + ".tmp").getAbsoluteFile();
        File parent = finalFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create shard parquet dir: " + parent.getAbsolutePath());
        }
        this.groupFactory = new SimpleGroupFactory(SCHEMA);
        this.writer = ExampleParquetWriter.builder(new LocalOutputFile(tmpFile))
                .withConf(new Configuration(false))
                .withType(SCHEMA)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .build();
    }

    /**
     * 새 scalar shard parquet writer를 연다.
     */
    static ScalarShardParquetWriter open(String finalPath) throws IOException {
        return new ScalarShardParquetWriter(finalPath);
    }

    /**
     * feature 한 row를 parquet row 하나로 기록한다.
     *
     * <p>호출자는 이 메서드가 반환한 뒤 같은 byte 배열을 재사용해도 된다.
     * writer가 row를 쓰는 시점에 parquet 쪽으로 바로 밀어 넣기 때문이다.
     */
    void writeRow(int featureId, int valueLen, byte[] valuesBlob, byte[] validBlob) throws IOException {
        Group row = groupFactory.newGroup()
                .append("feature_id", featureId)
                .append("value_len", valueLen)
                .append("values_blob", Binary.fromConstantByteArray(valuesBlob))
                .append("valid_blob", Binary.fromConstantByteArray(validBlob));
        writer.write(row);
    }

    /**
     * 닫을 때까지 row를 하나도 쓰지 않으면 빈 shard parquet가 만들어진다.
     */
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

    /**
     * parquet writer가 요구하는 OutputFile 인터페이스를 로컬 파일로 연결한다.
     *
     * <p>여기서는 하둡 파일시스템 추상화가 필요 없으므로,
     * 단순히 FileOutputStream을 PositionOutputStream으로 감싼다.
     */
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

    /**
     * parquet writer가 현재 파일 offset을 추적할 수 있게 하는 OutputStream wrapper다.
     */
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
            pos += 1L;
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
