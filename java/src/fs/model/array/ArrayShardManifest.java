package fs.model.array;

import fs.model.common.PointColumnSpec;

import java.io.File;
import java.util.List;

/**
 * Array binary shard dataset 전체 구성을 설명하는 최상위 manifest다.
 */
public class ArrayShardManifest {
    public final int version;
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final int nFeatures;
    public final String shardPath;
    public final int nShards;
    public final int samplesPerBlock;
    public final int blocksPerFeature;
    public final String featureIdType;
    public final String flagsType;
    public final String offsetType;
    public final String defaultCodec;
    public final String endianness;
    public final String sampleKeyCol;
    public final String featureKeyCol;
    public final ArrayBinaryShardInfo[] shards;
    public final List<PointColumnSpec> pointSchema;

    public ArrayShardManifest(
            int version,
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            int nFeatures,
            String shardPath,
            int nShards,
            int samplesPerBlock,
            int blocksPerFeature,
            String featureIdType,
            String flagsType,
            String offsetType,
            String defaultCodec,
            String endianness,
            String sampleKeyCol,
            String featureKeyCol,
            ArrayBinaryShardInfo[] shards,
            List<PointColumnSpec> pointSchema) {
        this.version = version;
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.nFeatures = nFeatures;
        this.shardPath = shardPath;
        this.nShards = nShards;
        this.samplesPerBlock = samplesPerBlock;
        this.blocksPerFeature = blocksPerFeature;
        this.featureIdType = featureIdType;
        this.flagsType = flagsType;
        this.offsetType = offsetType;
        this.defaultCodec = defaultCodec;
        this.endianness = endianness;
        this.sampleKeyCol = sampleKeyCol;
        this.featureKeyCol = featureKeyCol;
        this.shards = (shards == null) ? new ArrayBinaryShardInfo[0] : shards;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
    }

    public ArrayBinaryShardInfo shardInfo(int shardId) {
        return shards[shardId];
    }

    public String blocksIndexPath(int shardId) {
        return new File(shardPath, shardInfo(shardId).blocksIndexName).getPath();
    }

    public String blocksDataPath(int shardId) {
        return new File(shardPath, shardInfo(shardId).blocksDataName).getPath();
    }
}
