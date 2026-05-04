package fs.io.array;

import fs.model.array.ArrayBinaryShardInfo;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;

/**
 * Array binary shard의 idx/bin 파일 포맷 상수와 저수준 입출력 helper를 모아둔 클래스다.
 */
public final class ArrayBinaryFormat {
    public static final int CODEC_NONE = 0;
    public static final int CODEC_ZSTD = 1;

    public static final int FILE_VERSION = 3;
    public static final String FILE_ENDIANNESS = "little";
    public static final String DEFAULT_CODEC_NAME = "none";
    public static final String DEFAULT_SAMPLE_KEY_COL = "sample_key";
    public static final String DEFAULT_FEATURE_KEY_COL = "feature_key";

    public static final byte[] BLOCKS_INDEX_MAGIC = "ABLOCKIX".getBytes(StandardCharsets.US_ASCII);
    public static final byte[] BLOCKS_DATA_MAGIC = "ABLOCKSB".getBytes(StandardCharsets.US_ASCII);

    public static final int FILE_HEADER_BYTES = 64;
    public static final int BLOCK_RECORD_BYTES = 32;
    public static final int BLOCK_PAYLOAD_HEADER_BYTES = 48;

    private ArrayBinaryFormat() {
    }

    public static String blocksIndexName(int shardId) {
        return String.format("shard_%04d.blocks.idx", shardId);
    }

    public static String blocksDataName(int shardId) {
        return String.format("shard_%04d.blocks.bin", shardId);
    }

    public static File blocksIndexFile(String shardPath, ArrayBinaryShardInfo shardInfo) {
        return new File(shardPath, shardInfo.blocksIndexName);
    }

    public static File blocksDataFile(String shardPath, ArrayBinaryShardInfo shardInfo) {
        return new File(shardPath, shardInfo.blocksDataName);
    }

    public static void writePlaceholderHeader(RandomAccessFile raf) throws IOException {
        raf.setLength(0L);
        raf.write(new byte[FILE_HEADER_BYTES]);
    }

    public static void writeFileHeader(RandomAccessFile raf, byte[] magic, int recordBytes, long entryCount, long auxCount, int shardId)
            throws IOException {
        writeFileHeader(raf, magic, FILE_VERSION, recordBytes, entryCount, auxCount, shardId);
    }

    public static void writeFileHeader(RandomAccessFile raf, byte[] magic, int version, int recordBytes, long entryCount, long auxCount, int shardId)
            throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(FILE_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bb.put(fixedMagic(magic));
        bb.putShort((short) version);
        bb.putShort((short) FILE_HEADER_BYTES);
        bb.putShort((short) recordBytes);
        bb.putShort((short) 0);
        bb.putLong(entryCount);
        bb.putLong(auxCount);
        bb.putInt(shardId);
        bb.position(FILE_HEADER_BYTES);
        raf.seek(0L);
        raf.write(bb.array());
    }

    public static FileHeader readFileHeader(File file, byte[] expectedMagic) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] raw = new byte[FILE_HEADER_BYTES];
            raf.readFully(raw);
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            byte[] magic = new byte[8];
            bb.get(magic);
            if (!equalMagic(magic, expectedMagic)) {
                throw new IOException("unexpected magic for " + file.getAbsolutePath());
            }
            int version = bb.getShort() & 0xFFFF;
            int headerBytes = bb.getShort() & 0xFFFF;
            int recordBytes = bb.getShort() & 0xFFFF;
            int flags = bb.getShort() & 0xFFFF;
            long entryCount = bb.getLong();
            long auxCount = bb.getLong();
            int shardId = bb.getInt();
            if (version != FILE_VERSION) {
                throw new IOException("unsupported version=" + version + " for " + file.getAbsolutePath());
            }
            if (headerBytes != FILE_HEADER_BYTES) {
                throw new IOException("unexpected header size=" + headerBytes + " for " + file.getAbsolutePath());
            }
            return new FileHeader(version, recordBytes, flags, entryCount, auxCount, shardId);
        }
    }

    public static void writeBlockRecord(
            RandomAccessFile raf,
            long dataOffset,
            long dataLength,
            long pointCount,
            int codecId) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(BLOCK_RECORD_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(dataOffset);
        bb.putLong(dataLength);
        bb.putLong(pointCount);
        bb.put((byte) codecId);
        bb.put((byte) 0);
        bb.putShort((short) 0);
        bb.putInt(0);
        raf.write(bb.array());
    }

    public static long writeBlockPayload(
            RandomAccessFile raf,
            int featureId,
            int blockId,
            long sampleIdStart,
            int sampleCount,
            long pointCount,
            byte[] sampleFlags,
            byte[] sampleOffsetsBlob,
            byte[] encodedColumnsBlob,
            int schemaColumnCount,
            int codecId) throws IOException {
        byte[] flags = (sampleFlags == null) ? new byte[0] : sampleFlags;
        byte[] offsets = (sampleOffsetsBlob == null) ? new byte[0] : sampleOffsetsBlob;
        byte[] encodedColumns = (encodedColumnsBlob == null) ? new byte[0] : encodedColumnsBlob;
        long start = raf.getFilePointer();
        ByteBuffer bb = ByteBuffer.allocate(BLOCK_PAYLOAD_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(featureId);
        bb.putInt(blockId);
        bb.putLong(sampleIdStart);
        bb.putInt(sampleCount);
        bb.put((byte) codecId);
        bb.put((byte) 0);
        bb.putShort((short) schemaColumnCount);
        bb.putLong(pointCount);
        bb.putInt(flags.length);
        bb.putInt(offsets.length);
        bb.putInt(encodedColumns.length);
        bb.putInt(0);
        raf.write(bb.array());
        raf.write(flags);
        raf.write(offsets);
        raf.write(encodedColumns);
        return raf.getFilePointer() - start;
    }

    public static BlockIndexRecord readBlockRecord(FileChannel channel, long rowInShard) throws IOException {
        long position = FILE_HEADER_BYTES + (rowInShard * (long) BLOCK_RECORD_BYTES);
        ByteBuffer bb = ByteBuffer.allocate(BLOCK_RECORD_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, bb, position);
        bb.flip();
        long dataOffset = bb.getLong();
        long dataLength = bb.getLong();
        long pointCount = bb.getLong();
        int codec = bb.get() & 0xFF;
        int blockFlags = bb.get() & 0xFF;
        int reserved0 = bb.getShort() & 0xFFFF;
        long crc32 = bb.getInt() & 0xFFFFFFFFL;
        return new BlockIndexRecord(dataOffset, dataLength, pointCount, codec, blockFlags, reserved0, crc32);
    }

    public static byte[] readBytes(FileChannel channel, long position, int length) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(length);
        readFully(channel, bb, position);
        return bb.array();
    }

    private static void readFully(FileChannel channel, ByteBuffer bb, long position) throws IOException {
        long pos = position;
        while (bb.hasRemaining()) {
            int n = channel.read(bb, pos);
            if (n < 0) {
                throw new EOFException("unexpected EOF at position=" + pos);
            }
            pos += n;
        }
    }

    private static byte[] fixedMagic(byte[] magic) {
        byte[] out = new byte[8];
        System.arraycopy(magic, 0, out, 0, Math.min(out.length, magic.length));
        return out;
    }

    private static boolean equalMagic(byte[] actual, byte[] expected) {
        if (actual.length != 8) {
            return false;
        }
        byte[] want = fixedMagic(expected);
        for (int i = 0; i < 8; i++) {
            if (actual[i] != want[i]) {
                return false;
            }
        }
        return true;
    }

    public static final class FileHeader {
        public final int version;
        public final int recordBytes;
        public final int flags;
        public final long entryCount;
        public final long auxCount;
        public final int shardId;

        FileHeader(int version, int recordBytes, int flags, long entryCount, long auxCount, int shardId) {
            this.version = version;
            this.recordBytes = recordBytes;
            this.flags = flags;
            this.entryCount = entryCount;
            this.auxCount = auxCount;
            this.shardId = shardId;
        }
    }

    public static final class BlockIndexRecord {
        public final long dataOffset;
        public final long dataLength;
        public final long pointCount;
        public final int codec;
        public final int blockFlags;
        public final int reserved0;
        public final long crc32Optional;

        BlockIndexRecord(
                long dataOffset,
                long dataLength,
                long pointCount,
                int codec,
                int blockFlags,
                int reserved0,
                long crc32Optional) {
            this.dataOffset = dataOffset;
            this.dataLength = dataLength;
            this.pointCount = pointCount;
            this.codec = codec;
            this.blockFlags = blockFlags;
            this.reserved0 = reserved0;
            this.crc32Optional = crc32Optional;
        }
    }
}
