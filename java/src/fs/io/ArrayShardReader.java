package fs.io;

import fs.model.ArrayBlockLocation;
import fs.model.ArrayFeatureBlock;
import fs.model.ArrayShardManifest;
import fs.model.ArrayTrace;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ArrayShardReader implements AutoCloseable {
    private final ArrayShardManifest manifest;
    private final ArrayBinaryShardReader binaryReader;

    public ArrayShardReader(ArrayShardManifest manifest) {
        this.manifest = manifest;
        this.binaryReader = new ArrayBinaryShardReader(manifest);
    }

    public ArrayFeatureBlock loadBlock(int shardId, int rowInShard) throws IOException {
        return binaryReader.loadBlock(shardId, rowInShard);
    }

    public Map<Long, ArrayTrace> loadFeatureSamples(int featureId, long[] sampleIds, ArrayFeatureLocatorIndex locatorIndex) throws IOException {
        LinkedHashMap<Long, ArrayTrace> out = new LinkedHashMap<Long, ArrayTrace>();
        if (sampleIds == null || sampleIds.length == 0) {
            return out;
        }
        for (long sampleId : sampleIds) {
            out.put(sampleId, new ArrayTrace(sampleId, (byte) 0, new double[0], new double[0]));
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
            ArrayFeatureBlock block = loadBlock(loc.shardId, loc.rowInShard);
            for (long sampleId : entry.getValue()) {
                ArrayTrace trace = block.traceForSampleId(sampleId);
                if (trace != null) {
                    out.put(sampleId, trace);
                }
            }
        }
        return out;
    }

    public Map<Long, ArrayTrace> loadFeatureSamplesBySampleIds(
            int featureId,
            long[] sampleIds,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
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
        Map<Long, ArrayTrace> tracesBySampleId = loadFeatureSamples(featureId, denseSampleIdArray, locatorIndex);
        for (long sampleId : sampleIds) {
            Long denseSampleId = idx.findSampleId(sampleId);
            if (denseSampleId == null) {
                out.put(sampleId, new ArrayTrace(-1L, (byte) 0, new double[0], new double[0]));
            } else {
                ArrayTrace trace = tracesBySampleId.get(denseSampleId);
                if (trace == null) {
                    trace = new ArrayTrace(denseSampleId, (byte) 0, new double[0], new double[0]);
                }
                out.put(sampleId, trace);
            }
        }
        return out;
    }

    public Map<String, ArrayTrace> loadFeatureSamplesBySampleKeys(
            int featureId,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
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
        Map<Long, ArrayTrace> tracesBySampleId = loadFeatureSamples(featureId, requestedSampleIds, locatorIndex);
        for (String sampleKey : sampleKeys) {
            Long sampleId = idx.findSampleIdByKey(sampleKey);
            if (sampleId == null) {
                out.put(sampleKey, new ArrayTrace(-1L, (byte) 0, new double[0], new double[0]));
            } else {
                ArrayTrace trace = tracesBySampleId.get(sampleId);
                if (trace == null) {
                    trace = new ArrayTrace(sampleId, (byte) 0, new double[0], new double[0]);
                }
                out.put(sampleKey, trace);
            }
        }
        return out;
    }

    public Map<String, ArrayTrace> loadFeatureSamplesByKeys(
            String featureKey,
            String[] sampleKeys,
            ArrayFeatureLocatorIndex locatorIndex,
            ArrayFeatureIdIndex featureIdIndex,
            ArraySampleIdIndex sampleIdIndex) throws Exception {
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
                    out.put(sampleKey, new ArrayTrace(-1L, (byte) 0, new double[0], new double[0]));
                }
            }
            return out;
        }
        return loadFeatureSamplesBySampleKeys(featureId.intValue(), sampleKeys, locatorIndex, sampleIdIndex);
    }

    public static Map<Long, ArrayTrace> loadFeatureSamplesBySampleIds(
            ArrayShardManifest manifest,
            int featureId,
            long[] sampleIds) throws Exception {
        ArrayFeatureLocatorIndex locatorIndex = ArrayFeatureLocatorIndex.load(manifest);
        ArraySampleIdIndex sampleIdIndex = ArraySampleIdIndex.load(manifest.sampleMetaPath);
        try (ArrayShardReader reader = new ArrayShardReader(manifest)) {
            return reader.loadFeatureSamplesBySampleIds(featureId, sampleIds, locatorIndex, sampleIdIndex);
        }
    }

    public static Map<String, ArrayTrace> loadFeatureSamplesByKeys(
            ArrayShardManifest manifest,
            String featureKey,
            String[] sampleKeys) throws Exception {
        ArrayFeatureLocatorIndex locatorIndex = ArrayFeatureLocatorIndex.load(manifest);
        ArrayFeatureIdIndex featureIdIndex = ArrayFeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
        ArraySampleIdIndex sampleIdIndex = ArraySampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
        try (ArrayShardReader reader = new ArrayShardReader(manifest)) {
            return reader.loadFeatureSamplesByKeys(featureKey, sampleKeys, locatorIndex, featureIdIndex, sampleIdIndex);
        }
    }

    @Override
    public void close() throws IOException {
        binaryReader.close();
    }
}
