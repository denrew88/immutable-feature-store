package fs.io;

import fs.model.ArrayBinaryShardInfo;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;

final class ArrayBinaryFormat {
    static final int CODEC_NONE = 0;
    static final int CODEC_ZSTD = 1;

    static final int FILE_VERSION = 2;
    static final String FILE_ENDIANNESS = "little";
    static final String DEFAULT_CODEC_NAME = "none";
    static final String DEFAULT_SAMPLE_KEY_COL = "sample_key";
    static final String DEFAULT_FEATURE_KEY_COL = "feature_key";

    static final byte[] BLOCKS_INDEX_MAGIC = "ABLOCKIX".getBytes(StandardCharsets.US_ASCII);
    static final byte[] BLOCKS_DATA_MAGIC = "ABLOCKSB".getBytes(StandardCharsets.US_ASCII);

    static final int FILE_HEADER_BYTES = 64;
    static final int BLOCK_RECORD_BYTES = 32;
    static final int BLOCK_PAYLOAD_HEADER_BYTES = 48;

    private ArrayBinaryFormat() {
    }

    static String blocksIndexName(int shardId) {
        return String.format("shard_%04d.blocks.idx", shardId);
    }

    static String blocksDataName(int shardId) {
        return String.format("shard_%04d.blocks.bin", shardId);
    }

    static File blocksIndexFile(String shardPath, ArrayBinaryShardInfo shardInfo) {
        return new File(shardPath, shardInfo.blocksIndexName);
    }

    static File blocksDataFile(String shardPath, ArrayBinaryShardInfo shardInfo) {
        return new File(shardPath, shardInfo.blocksDataName);
    }

    static void writePlaceholderHeader(RandomAccessFile raf) throws IOException {
        raf.setLength(0L);
        raf.write(new byte[FILE_HEADER_BYTES]);
    }

    static void writeFileHeader(RandomAccessFile raf, byte[] magic, int recordBytes, long entryCount, long auxCount, int shardId)
            throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(FILE_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bb.put(fixedMagic(magic));
        bb.putShort((short) FILE_VERSION);
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

    static FileHeader readFileHeader(File file, byte[] expectedMagic) throws IOException {
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
            return new FileHeader(recordBytes, flags, entryCount, auxCount, shardId);
        }
    }

    static void writeBlockRecord(
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

    static long writeBlockPayload(
            RandomAccessFile raf,
            int featureId,
            int blockId,
            long sampleIdStart,
            int sampleCount,
            long pointCount,
            byte[] sampleFlags,
            byte[] sampleOffsetsBlob,
            byte[] timeBlob,
            byte[] valueBlob,
            int codecId) throws IOException {
        byte[] flags = (sampleFlags == null) ? new byte[0] : sampleFlags;
        byte[] offsets = (sampleOffsetsBlob == null) ? new byte[0] : sampleOffsetsBlob;
        byte[] time = (timeBlob == null) ? new byte[0] : timeBlob;
        byte[] value = (valueBlob == null) ? new byte[0] : valueBlob;
        long start = raf.getFilePointer();
        ByteBuffer bb = ByteBuffer.allocate(BLOCK_PAYLOAD_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(featureId);
        bb.putInt(blockId);
        bb.putLong(sampleIdStart);
        bb.putInt(sampleCount);
        bb.put((byte) codecId);
        bb.put((byte) 0);
        bb.putShort((short) 0);
        bb.putLong(pointCount);
        bb.putInt(flags.length);
        bb.putInt(offsets.length);
        bb.putInt(time.length);
        bb.putInt(value.length);
        raf.write(bb.array());
        raf.write(flags);
        raf.write(offsets);
        raf.write(time);
        raf.write(value);
        return raf.getFilePointer() - start;
    }

    static BlockIndexRecord readBlockRecord(FileChannel channel, long rowInShard) throws IOException {
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

    static byte[] readBytes(FileChannel channel, long position, int length) throws IOException {
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

    static final class FileHeader {
        final int recordBytes;
        final int flags;
        final long entryCount;
        final long auxCount;
        final int shardId;

        FileHeader(int recordBytes, int flags, long entryCount, long auxCount, int shardId) {
            this.recordBytes = recordBytes;
            this.flags = flags;
            this.entryCount = entryCount;
            this.auxCount = auxCount;
            this.shardId = shardId;
        }
    }

    static final class BlockIndexRecord {
        final long dataOffset;
        final long dataLength;
        final long pointCount;
        final int codec;
        final int blockFlags;
        final int reserved0;
        final long crc32Optional;

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
