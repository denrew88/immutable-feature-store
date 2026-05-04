package fs.io.array;

import fs.model.array.ArrayBinaryShardInfo;
import fs.model.array.ArrayBlockLocation;
import fs.model.array.ArrayShardManifest;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Array feature가 어느 shard와 block 범위에 들어 있는지 찾기 위한 locator 인덱스다.
 */
public class ArrayFeatureLocatorIndex {
    private final ArrayShardManifest manifest;
    private final int[] featureToShard;

    private ArrayFeatureLocatorIndex(ArrayShardManifest manifest, int[] featureToShard) {
        this.manifest = manifest;
        this.featureToShard = featureToShard;
    }

    public static ArrayFeatureLocatorIndex load(ArrayShardManifest manifest) throws IOException {
        int[] featureToShard = new int[manifest.nFeatures];
        for (int i = 0; i < featureToShard.length; i++) {
            featureToShard[i] = -1;
        }
        for (int shardId = 0; shardId < manifest.nShards; shardId++) {
            ArrayBinaryShardInfo shardInfo = manifest.shardInfo(shardId);
            for (int featureId = shardInfo.featureIdStart; featureId <= shardInfo.featureIdEnd; featureId++) {
                if (featureId < 0 || featureId >= manifest.nFeatures) {
                    throw new IOException("feature_id out of manifest range: " + featureId);
                }
                featureToShard[featureId] = shardId;
            }
        }
        return new ArrayFeatureLocatorIndex(manifest, featureToShard);
    }

    public List<ArrayBlockLocation> blocksForFeature(int featureId) {
        int shardId = shardForFeature(featureId);
        if (shardId < 0) {
            return Collections.emptyList();
        }
        ArrayBinaryShardInfo shardInfo = manifest.shardInfo(shardId);
        int localFeature = featureId - shardInfo.featureIdStart;
        ArrayList<ArrayBlockLocation> out = new ArrayList<ArrayBlockLocation>(manifest.blocksPerFeature);
        for (int blockId = 0; blockId < manifest.blocksPerFeature; blockId++) {
            long sampleIdStart = ((long) blockId) * (long) manifest.samplesPerBlock;
            int sampleCount = sampleCountForBlock(blockId);
            if (sampleCount <= 0) {
                break;
            }
            int rowInShard = localFeature * manifest.blocksPerFeature + blockId;
            out.add(new ArrayBlockLocation(
                    featureId,
                    blockId,
                    shardId,
                    rowInShard,
                    sampleIdStart,
                    sampleIdStart + sampleCount - 1L));
        }
        return out;
    }

    public ArrayBlockLocation findBlockForSampleId(int featureId, long sampleId) {
        int shardId = shardForFeature(featureId);
        if (shardId < 0 || sampleId < 0L || sampleId >= manifest.nSamples) {
            return null;
        }
        int blockId = (int) (sampleId / manifest.samplesPerBlock);
        if (blockId < 0 || blockId >= manifest.blocksPerFeature) {
            return null;
        }
        ArrayBinaryShardInfo shardInfo = manifest.shardInfo(shardId);
        int localFeature = featureId - shardInfo.featureIdStart;
        long sampleIdStart = ((long) blockId) * (long) manifest.samplesPerBlock;
        int sampleCount = sampleCountForBlock(blockId);
        return new ArrayBlockLocation(
                featureId,
                blockId,
                shardId,
                localFeature * manifest.blocksPerFeature + blockId,
                sampleIdStart,
                sampleIdStart + sampleCount - 1L);
    }

    private int shardForFeature(int featureId) {
        if (featureId < 0 || featureId >= featureToShard.length) {
            return -1;
        }
        return featureToShard[featureId];
    }

    private int sampleCountForBlock(int blockId) {
        long start = ((long) blockId) * (long) manifest.samplesPerBlock;
        long remaining = (long) manifest.nSamples - start;
        if (remaining <= 0L) {
            return 0;
        }
        return (int) Math.min((long) manifest.samplesPerBlock, remaining);
    }
}
