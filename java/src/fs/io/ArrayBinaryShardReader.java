package fs.io;

import fs.model.ArrayBinaryShardInfo;
import fs.model.ArrayFeatureBlock;
import fs.model.ArrayShardManifest;
import fs.model.LogicalType;
import fs.model.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ArrayBinaryShardReader implements AutoCloseable {
    private final ArrayShardManifest manifest;
    private final HashMap<Integer, CachedShard> shardCache;
    private final HashMap<String, HashMap<Long, String>> dictionaryCache;

    public ArrayBinaryShardReader(ArrayShardManifest manifest) {
        this.manifest = manifest;
        this.shardCache = new HashMap<Integer, CachedShard>();
        this.dictionaryCache = new HashMap<String, HashMap<Long, String>>();
    }

    public List<PointColumnSpec> pointSchema() {
        return manifest.pointSchema;
    }

    public ArrayFeatureBlock loadBlock(int shardId, int rowInShard) throws IOException {
        return loadBlock(shardId, rowInShard, false);
    }

    public ArrayFeatureBlock loadBlock(int shardId, int rowInShard, boolean decodeCategorical) throws IOException {
        CachedShard shard = shard(shardId);
        ArrayBinaryShardInfo shardInfo = manifest.shardInfo(shardId);
        if (rowInShard < 0 || rowInShard >= shardInfo.blockCount) {
            throw new IOException("rowInShard out of range: shard=" + shardId + " row=" + rowInShard);
        }

        int localFeature = rowInShard / manifest.blocksPerFeature;
        int blockId = rowInShard % manifest.blocksPerFeature;
        if (localFeature < 0 || localFeature >= shardInfo.featureCount) {
            throw new IOException("invalid local feature for shard=" + shardId + " row=" + rowInShard);
        }

        int featureId = shardInfo.featureIdStart + localFeature;
        long sampleIdStart = ((long) blockId) * (long) manifest.samplesPerBlock;
        int sampleCount = sampleCountForBlock(blockId);

        ArrayBinaryFormat.BlockIndexRecord record = ArrayBinaryFormat.readBlockRecord(shard.blocksIndexChannel, rowInShard);
        ArrayFeatureBlock block;
        if (record.dataLength == 0L) {
            block = emptyBlock(featureId, blockId, sampleIdStart, sampleCount, decodeCategorical);
        } else {
            if (record.codec != ArrayBinaryFormat.CODEC_NONE) {
                throw new IOException("unsupported codec id=" + record.codec + " for shard_id=" + shardId);
            }
            byte[] payload = ArrayBinaryFormat.readBytes(shard.blocksDataChannel, record.dataOffset, (int) record.dataLength);
            block = decodePayload(featureId, blockId, sampleIdStart, sampleCount, record, payload, decodeCategorical);
        }
        return block;
    }

    private ArrayFeatureBlock decodePayload(
            int expectedFeatureId,
            int expectedBlockId,
            long expectedSampleIdStart,
            int expectedSampleCount,
            ArrayBinaryFormat.BlockIndexRecord record,
            byte[] payload,
            boolean decodeCategorical) throws IOException {
        if (payload.length < ArrayBinaryFormat.BLOCK_PAYLOAD_HEADER_BYTES) {
            throw new IOException("payload too short: " + payload.length);
        }
        ByteBuffer bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        int featureId = bb.getInt();
        int blockId = bb.getInt();
        long sampleIdStart = bb.getLong();
        int sampleCount = bb.getInt();
        int codec = bb.get() & 0xFF;
        int headerFlags = bb.get() & 0xFF;
        int schemaColumnCount = bb.getShort() & 0xFFFF;
        long pointCount = bb.getLong();
        int flagsBytes = bb.getInt();
        int offsetsBytes = bb.getInt();
        int encodedColumnsOrTimeBytes = bb.getInt();
        int valueBytesOrReserved = bb.getInt();

        if (featureId != expectedFeatureId || blockId != expectedBlockId) {
            throw new IOException("payload/index mismatch");
        }
        if (sampleIdStart != expectedSampleIdStart || sampleCount != expectedSampleCount) {
            throw new IOException("payload sample range mismatch");
        }
        if (pointCount != record.pointCount || codec != record.codec) {
            throw new IOException("payload metadata mismatch");
        }
        if (codec != ArrayBinaryFormat.CODEC_NONE) {
            throw new IOException("unsupported codec id=" + codec);
        }

        byte[] sampleFlags = new byte[sampleCount];
        bb.get(sampleFlags);
        byte[] sampleOffsetsBlob = new byte[offsetsBytes];
        bb.get(sampleOffsetsBlob);
        long[] sampleOffsets = ArrayUtils.decodeLongArray(sampleOffsetsBlob);
        if (sampleOffsets.length != sampleCount + 1) {
            throw new IOException("invalid sample_offsets length=" + sampleOffsets.length);
        }
        if (sampleOffsets[sampleOffsets.length - 1] != pointCount) {
            throw new IOException("point_count/sample_offsets mismatch");
        }

        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
        if (manifest.version == ArrayBinaryFormat.LEGACY_FILE_VERSION) {
            int expectedLength = ArrayBinaryFormat.BLOCK_PAYLOAD_HEADER_BYTES + flagsBytes + offsetsBytes + encodedColumnsOrTimeBytes + valueBytesOrReserved;
            if (expectedLength != payload.length) {
                throw new IOException("payload length mismatch: expected=" + expectedLength + " got=" + payload.length);
            }
            byte[] timeBlob = new byte[encodedColumnsOrTimeBytes];
            bb.get(timeBlob);
            byte[] valueBlob = new byte[valueBytesOrReserved];
            bb.get(valueBlob);
            columns.put("time", ArrayUtils.decodeDoubleArray(timeBlob, (int) pointCount));
            columns.put("value", ArrayUtils.decodeDoubleArray(valueBlob, (int) pointCount));
        } else {
            if (schemaColumnCount != manifest.pointSchema.size()) {
                throw new IOException("payload point_schema mismatch: expected=" + manifest.pointSchema.size() + " got=" + schemaColumnCount);
            }
            int expectedLength = ArrayBinaryFormat.BLOCK_PAYLOAD_HEADER_BYTES + flagsBytes + offsetsBytes + encodedColumnsOrTimeBytes;
            if (expectedLength != payload.length) {
                throw new IOException("payload length mismatch: expected=" + expectedLength + " got=" + payload.length);
            }
            byte[] encodedColumns = new byte[encodedColumnsOrTimeBytes];
            bb.get(encodedColumns);
            int cursor = 0;
            for (PointColumnSpec spec : manifest.pointSchema) {
                int byteCount = ArrayUtils.pointColumnBytes(spec, (int) pointCount);
                byte[] blob = new byte[byteCount];
                System.arraycopy(encodedColumns, cursor, blob, 0, byteCount);
                cursor += byteCount;
                Object values = ArrayUtils.decodePointColumn(blob, pointCount, spec);
                if (decodeCategorical && spec.logicalType == LogicalType.CATEGORICAL && spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty()) {
                    values = ArrayUtils.decodeCategoricalLabels(values, loadDictionary(spec.dictionaryPath));
                }
                columns.put(spec.name, values);
            }
            if (cursor != encodedColumns.length) {
                throw new IOException("decoded point column bytes mismatch: expected=" + encodedColumns.length + " got=" + cursor);
            }
        }
        return new ArrayFeatureBlock(
                featureId,
                blockId,
                sampleIdStart,
                sampleCount,
                pointCount,
                sampleFlags,
                sampleOffsets,
                columns);
    }

    private ArrayFeatureBlock emptyBlock(int featureId, int blockId, long sampleIdStart, int sampleCount, boolean decodeCategorical) {
        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
        List<PointColumnSpec> schema = manifest.pointSchema;
        for (PointColumnSpec spec : schema) {
            columns.put(spec.name, ArrayUtils.emptyPointColumn(spec, decodeCategorical && spec.logicalType == LogicalType.CATEGORICAL));
        }
        return new ArrayFeatureBlock(
                featureId,
                blockId,
                sampleIdStart,
                sampleCount,
                0L,
                new byte[sampleCount],
                new long[sampleCount + 1],
                columns);
    }

    private int sampleCountForBlock(int blockId) {
        long start = ((long) blockId) * (long) manifest.samplesPerBlock;
        long remaining = (long) manifest.nSamples - start;
        if (remaining <= 0L) {
            return 0;
        }
        return (int) Math.min((long) manifest.samplesPerBlock, remaining);
    }

    private HashMap<Long, String> loadDictionary(String dictionaryPath) throws IOException {
        HashMap<Long, String> cached = dictionaryCache.get(dictionaryPath);
        if (cached != null) {
            return cached;
        }
        HashMap<Long, String> out = new HashMap<Long, String>();
        try (Connection conn = DuckDBUtils.connect(null)) {
            String sql = "SELECT code, label FROM read_parquet(" + DuckDBUtils.quotePath(dictionaryPath) + ") ORDER BY code";
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery(sql)) {
                while (rs.next()) {
                    out.put(rs.getLong(1), rs.getString(2));
                }
            }
        } catch (Exception e) {
            throw new IOException("failed to load categorical dictionary: " + dictionaryPath, e);
        }
        dictionaryCache.put(dictionaryPath, out);
        return out;
    }

    private CachedShard shard(int shardId) throws IOException {
        CachedShard out = shardCache.get(shardId);
        if (out != null) {
            return out;
        }
        ArrayBinaryShardInfo shardInfo = manifest.shardInfo(shardId);
        File blocksIndexFile = ArrayBinaryFormat.blocksIndexFile(manifest.shardPath, shardInfo);
        File blocksDataFile = ArrayBinaryFormat.blocksDataFile(manifest.shardPath, shardInfo);
        ArrayBinaryFormat.FileHeader indexHeader = ArrayBinaryFormat.readFileHeader(blocksIndexFile, ArrayBinaryFormat.BLOCKS_INDEX_MAGIC);
        ArrayBinaryFormat.FileHeader dataHeader = ArrayBinaryFormat.readFileHeader(blocksDataFile, ArrayBinaryFormat.BLOCKS_DATA_MAGIC);
        if (indexHeader.recordBytes != ArrayBinaryFormat.BLOCK_RECORD_BYTES) {
            throw new IOException("unexpected blocks.idx record size=" + indexHeader.recordBytes);
        }
        CachedShard created = new CachedShard(
                new RandomAccessFile(blocksIndexFile, "r"),
                new RandomAccessFile(blocksDataFile, "r"),
                indexHeader,
                dataHeader);
        shardCache.put(shardId, created);
        return created;
    }

    @Override
    public void close() throws IOException {
        IOException first = null;
        for (CachedShard shard : shardCache.values()) {
            try {
                shard.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                }
            }
        }
        shardCache.clear();
        dictionaryCache.clear();
        if (first != null) {
            throw first;
        }
    }

    private static final class CachedShard {
        final RandomAccessFile blocksIndex;
        final RandomAccessFile blocksData;
        final FileChannel blocksIndexChannel;
        final FileChannel blocksDataChannel;
        final ArrayBinaryFormat.FileHeader indexHeader;
        final ArrayBinaryFormat.FileHeader dataHeader;

        CachedShard(
                RandomAccessFile blocksIndex,
                RandomAccessFile blocksData,
                ArrayBinaryFormat.FileHeader indexHeader,
                ArrayBinaryFormat.FileHeader dataHeader) {
            this.blocksIndex = blocksIndex;
            this.blocksData = blocksData;
            this.blocksIndexChannel = blocksIndex.getChannel();
            this.blocksDataChannel = blocksData.getChannel();
            this.indexHeader = indexHeader;
            this.dataHeader = dataHeader;
        }

        void close() throws IOException {
            blocksIndex.close();
            blocksData.close();
        }
    }
}
