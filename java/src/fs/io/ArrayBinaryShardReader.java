package fs.io;

import fs.io.array.ArrayBinaryFormat;
import fs.io.array.ArrayFeatureIdIndex;
import fs.io.array.ArrayFeatureLocatorIndex;
import fs.io.array.ArraySampleIdIndex;
import fs.io.common.ArrayUtils;
import fs.io.common.JsonUtils;
import fs.model.array.ArrayBinaryShardInfo;
import fs.model.array.ArrayBlockLocation;
import fs.model.array.ArrayFeatureBlock;
import fs.model.array.ArrayShardManifest;
import fs.model.array.ArrayTrace;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * array binary shard를 읽는 low-level reader이다.
 *
 * <p>이 reader는 shard manifest와 locator index를 바탕으로
 * {@code (feature_id, block_id)} 조합에서 blocks.idx record를 찾고,
 * 그 record를 통해 blocks.bin payload를 찾는 경로를 따라 block을 로드한다.
 * 필요하면 block 안에서 sample trace를 다시 잘라 public 객체로 돌려준다.
 */
public class ArrayBinaryShardReader implements AutoCloseable {
    private final ArrayShardManifest manifest;
    private final HashMap<Integer, CachedShard> shardCache;
    private final HashMap<String, HashMap<Long, String>> dictionaryCache;

    /**
     * manifest 기반 reader를 생성한다.
     */
    public ArrayBinaryShardReader(ArrayShardManifest manifest) {
        this.manifest = manifest;
        this.shardCache = new HashMap<Integer, CachedShard>();
        this.dictionaryCache = new HashMap<String, HashMap<Long, String>>();
    }

    /**
     * reader가 해석 중인 point schema를 반환한다.
     */
    public List<PointColumnSpec> pointSchema() {
        return manifest.pointSchema;
    }

    /**
     * shard 안의 block row 하나를 로드한다.
     */
    public ArrayFeatureBlock loadBlock(int shardId, int rowInShard) throws IOException {
        return loadBlock(shardId, rowInShard, false);
    }

    /**
     * shard 안의 block row 하나를 로드한다.
     *
     * <p>{@code rowInShard}는 shard 안의 선형 row index이다.
     * reader는 이 값을 다시
     * {@code localFeatureIndex = rowInShard / blocksPerFeature},
     * {@code blockId = rowInShard % blocksPerFeature}
     * 로 해석해서 payload header와 대조한다.
     */
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

    /**
     * dense feature id와 dense sample id 배열을 기준으로 trace를 읽는다.
     *
     * <p>이 계열에서 가장 직접적인 조회 경로이다.
     * sample과 feature가 이미 내부 dense id 체계라고 가정하고, locator를 이용해 필요한 block만 읽는다.
     */
    public Map<Long, ArrayTrace> loadFeatureSamples(int featureId, long[] sampleIds, ArrayFeatureLocatorIndex locatorIndex) throws IOException {
        return loadFeatureSamples(featureId, sampleIds, locatorIndex, false);
    }

    /**
     * dense feature id와 dense sample id 배열을 기준으로 trace를 읽는다.
     *
     * <p>reader는 먼저 요청 sample들을 block 단위로 묶고, block은 한 번만 로드한 뒤
     * 그 안에서 sample trace를 다시 꺼낸다. 그래서 sample들이 같은 block에 모여 있을수록 유리하다.
     */
    public Map<Long, ArrayTrace> loadFeatureSamples(
            int featureId,
            long[] sampleIds,
            ArrayFeatureLocatorIndex locatorIndex,
            boolean decodeCategorical) throws IOException {
        LinkedHashMap<Long, ArrayTrace> out = new LinkedHashMap<Long, ArrayTrace>();
        if (sampleIds == null || sampleIds.length == 0) {
            return out;
        }
        for (long sampleId : sampleIds) {
            out.put(sampleId, emptyTrace(sampleId, decodeCategorical));
        }

        List<ArrayBlockLocation> blocks = locatorIndex.blocksForFeature(featureId);
        if (blocks.isEmpty()) {
            return out;
        }

        LinkedHashMap<String, List<Long>> sampleIdsByBlock = new LinkedHashMap<String, List<Long>>();
        LinkedHashMap<String, ArrayBlockLocation> blockByKey = new LinkedHashMap<String, ArrayBlockLocation>();
        for (long sampleId : sampleIds) {
            ArrayBlockLocation loc = locatorIndex.findBlockForSampleId(featureId, sampleId);
            if (loc == null) {
                continue;
            }
            String key = loc.shardId + ":" + loc.rowInShard;
            List<Long> ids = sampleIdsByBlock.get(key);
            if (ids == null) {
                ids = new ArrayList<Long>();
                sampleIdsByBlock.put(key, ids);
                blockByKey.put(key, loc);
            }
            ids.add(sampleId);
        }

        for (Map.Entry<String, List<Long>> entry : sampleIdsByBlock.entrySet()) {
            ArrayBlockLocation loc = blockByKey.get(entry.getKey());
            ArrayFeatureBlock block = loadBlock(loc.shardId, loc.rowInShard, decodeCategorical);
            for (long sampleId : entry.getValue()) {
                ArrayTrace trace = block.traceForSampleId(sampleId);
                if (trace != null) {
                    out.put(sampleId, trace);
                }
            }
        }
        return out;
    }

    /**
     * 외부 sample id 목록을 dense sample id로 변환한 뒤 trace를 읽는다.
     *
     * <p>{@link #loadFeatureSamples(int, long[], ArrayFeatureLocatorIndex, boolean)}와 최종 동작은 같지만,
     * 먼저 sample id index를 사용해 외부 id를 내부 dense sample id로 바꾸는 단계가 추가된다.
     */
    public Map<Long, ArrayTrace> loadFeatureSamplesBySampleIds(
            int featureId,
            long[] sampleIds,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
        return loadFeatureSamplesBySampleIds(featureId, sampleIds, locatorIndex, sampleIdIndex, false);
    }

    /**
     * 외부 sample id 목록을 dense sample id로 변환한 뒤 trace를 읽는다.
     */
    public Map<Long, ArrayTrace> loadFeatureSamplesBySampleIds(
            int featureId,
            long[] sampleIds,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex,
            boolean decodeCategorical) throws Exception {
        LinkedHashMap<Long, ArrayTrace> out = new LinkedHashMap<Long, ArrayTrace>();
        if (sampleIds == null || sampleIds.length == 0) {
            return out;
        }
        ArraySampleIdIndex idx = sampleIdIndex;
        if (idx == null) {
            if (manifest.sampleMetaPath == null || manifest.sampleMetaPath.isEmpty()) {
                throw new IllegalArgumentException("sampleMetaPath is required to resolve sample ids");
            }
            idx = ArraySampleIdIndex.load(manifest.sampleMetaPath);
        }

        List<Long> denseSampleIds = new ArrayList<Long>();
        for (long sampleId : sampleIds) {
            Long denseSampleId = idx.findSampleId(sampleId);
            if (denseSampleId != null) {
                denseSampleIds.add(denseSampleId);
            }
        }
        long[] denseSampleIdArray = new long[denseSampleIds.size()];
        for (int i = 0; i < denseSampleIds.size(); i++) {
            denseSampleIdArray[i] = denseSampleIds.get(i);
        }
        Map<Long, ArrayTrace> tracesBySampleId = loadFeatureSamples(featureId, denseSampleIdArray, locatorIndex, decodeCategorical);
        for (long sampleId : sampleIds) {
            Long denseSampleId = idx.findSampleId(sampleId);
            if (denseSampleId == null) {
                out.put(sampleId, emptyTrace(-1L, decodeCategorical));
            } else {
                ArrayTrace trace = tracesBySampleId.get(denseSampleId);
                if (trace == null) {
                    trace = emptyTrace(denseSampleId, decodeCategorical);
                }
                out.put(sampleId, trace);
            }
        }
        return out;
    }

    /**
     * sample key 목록을 dense sample id로 변환한 뒤 trace를 읽는다.
     *
     * <p>{@code loadFeatureSamplesBySampleIds(...)}와 비슷하지만, sample 쪽 입력이 id가 아니라 key라는 점이 다르다.
     * 즉 feature는 dense id로 받고, sample은 key를 dense id로 해석한 뒤 조회한다.
     */
    public Map<String, ArrayTrace> loadFeatureSamplesBySampleKeys(
            int featureId,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
        return loadFeatureSamplesBySampleKeys(featureId, sampleKeys, locatorIndex, sampleIdIndex, false);
    }

    /**
     * sample key 목록을 dense sample id로 변환한 뒤 trace를 읽는다.
     */
    public Map<String, ArrayTrace> loadFeatureSamplesBySampleKeys(
            int featureId,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex,
            boolean decodeCategorical) throws Exception {
        LinkedHashMap<String, ArrayTrace> out = new LinkedHashMap<String, ArrayTrace>();
        if (sampleKeys == null || sampleKeys.length == 0) {
            return out;
        }
        ArraySampleIdIndex idx = sampleIdIndex;
        if (idx == null) {
            if (manifest.sampleMetaPath == null || manifest.sampleMetaPath.isEmpty()) {
                throw new IllegalArgumentException("sampleMetaPath is required to resolve sample keys");
            }
            idx = ArraySampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
        }

        long[] denseSampleIds = new long[sampleKeys.length];
        int denseCount = 0;
        for (String sampleKey : sampleKeys) {
            Long sampleId = idx.findSampleIdByKey(sampleKey);
            if (sampleId != null) {
                denseSampleIds[denseCount++] = sampleId;
            }
        }
        long[] requestedSampleIds = new long[denseCount];
        System.arraycopy(denseSampleIds, 0, requestedSampleIds, 0, denseCount);
        Map<Long, ArrayTrace> tracesBySampleId = loadFeatureSamples(featureId, requestedSampleIds, locatorIndex, decodeCategorical);
        for (String sampleKey : sampleKeys) {
            Long sampleId = idx.findSampleIdByKey(sampleKey);
            if (sampleId == null) {
                out.put(sampleKey, emptyTrace(-1L, decodeCategorical));
            } else {
                ArrayTrace trace = tracesBySampleId.get(sampleId);
                if (trace == null) {
                    trace = emptyTrace(sampleId, decodeCategorical);
                }
                out.put(sampleKey, trace);
            }
        }
        return out;
    }

    /**
     * feature key와 sample key 조합으로 trace를 읽는다.
     *
     * <p>이 계열에서 가장 바깥쪽 API이다.
     * feature와 sample을 모두 key로 받기 때문에 외부 코드에서 쓰기 쉽고,
     * 내부적으로는 feature key와 sample key를 각각 dense id로 해석한 뒤 lower-level 경로로 위임한다.
     */
    public Map<String, ArrayTrace> loadFeatureSamplesByKeys(
            String featureKey,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArrayFeatureIdIndex featureIdIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
        return loadFeatureSamplesByKeys(featureKey, sampleKeys, locatorIndex, featureIdIndex, sampleIdIndex, false);
    }

    /**
     * feature key와 sample key 조합으로 trace를 읽는다.
     */
    public Map<String, ArrayTrace> loadFeatureSamplesByKeys(
            String featureKey,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArrayFeatureIdIndex featureIdIndex,
            ArraySampleIdIndex sampleIdIndex,
            boolean decodeCategorical) throws Exception {
        ArrayFeatureIdIndex features = featureIdIndex;
        if (features == null) {
            if (manifest.featureMetaPath == null || manifest.featureMetaPath.isEmpty()) {
                throw new IllegalArgumentException("featureMetaPath is required to resolve feature keys");
            }
            features = ArrayFeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
        }
        Integer featureId = features.findFeatureIdByKey(featureKey);
        if (featureId == null) {
            LinkedHashMap<String, ArrayTrace> out = new LinkedHashMap<String, ArrayTrace>();
            if (sampleKeys != null) {
                for (String sampleKey : sampleKeys) {
                    out.put(sampleKey, emptyTrace(-1L, decodeCategorical));
                }
            }
            return out;
        }
        return loadFeatureSamplesBySampleKeys(featureId.intValue(), sampleKeys, locatorIndex, sampleIdIndex, decodeCategorical);
    }

    /**
     * blocks.bin payload를 block 객체로 디코드한다.
     *
     * <p>payload header와 blocks.idx record가 서로 맞는지 먼저 검증한 뒤,
     * sample flags, sample offsets, point column blob들을 순서대로 풀어서
     * {@link ArrayFeatureBlock}을 만든다.
     */
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

        if (schemaColumnCount != manifest.pointSchema.size()) {
            throw new IOException("payload point_schema mismatch: expected=" + manifest.pointSchema.size() + " got=" + schemaColumnCount);
        }
        int expectedLength = ArrayBinaryFormat.BLOCK_PAYLOAD_HEADER_BYTES + flagsBytes + offsetsBytes + encodedColumnsOrTimeBytes;
        if (expectedLength != payload.length) {
            throw new IOException("payload length mismatch: expected=" + expectedLength + " got=" + payload.length);
        }
        byte[] encodedColumns = new byte[encodedColumnsOrTimeBytes];
        bb.get(encodedColumns);
        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
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

    /**
     * categorical dictionary JSON을 캐시와 함께 로드한다.
     */
    private HashMap<Long, String> loadDictionary(String dictionaryPath) throws IOException {
        HashMap<Long, String> cached = dictionaryCache.get(dictionaryPath);
        if (cached != null) {
            return cached;
        }
        if (!dictionaryPath.toLowerCase().endsWith(".json")) {
            throw new IOException("categorical dictionary must be a JSON file: " + dictionaryPath);
        }
        HashMap<Long, String> out = JsonUtils.readCategoricalDictionary(dictionaryPath);
        dictionaryCache.put(dictionaryPath, out);
        return out;
    }

    /**
     * shard id에 해당하는 blocks.idx / blocks.bin 파일 핸들을 연다.
     *
     * <p>첫 번째 호출 때 shard를 cache에 올리고, 이후에는 재사용한다.
     */
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

    /**
     * trace가 없을 때 반환할 빈 trace 객체를 만든다.
     */
    private ArrayTrace emptyTrace(long sampleId, boolean decodeCategorical) {
        LinkedHashMap<String, Object> columns = new LinkedHashMap<String, Object>();
        for (PointColumnSpec spec : manifest.pointSchema) {
            boolean decode = decodeCategorical && spec.logicalType == LogicalType.CATEGORICAL;
            columns.put(spec.name, ArrayUtils.emptyPointColumn(spec, decode));
        }
        return new ArrayTrace(sampleId, (byte) 0, columns);
    }

    /**
     * 열린 shard 파일 핸들과 dictionary cache를 정리한다.
     */
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
