package fs.model;

import java.io.File;

public class ShardManifest {
    public final String sampleMetaPath;
    public final int nSamples;
    public final String shardPath;
    public final int nShards;
    public final String featureLocatorPath;
    public final String featureLocatorFormat;
    public final String featureIdType;
    public final String valuesType;
    public final String validType;
    public final String statsYCol;

    public ShardManifest(
            String sampleMetaPath,
            int nSamples,
            String shardPath,
            int nShards,
            String featureLocatorPath,
            String featureLocatorFormat,
            String featureIdType,
            String valuesType,
            String validType,
            String statsYCol) {
        this.sampleMetaPath = sampleMetaPath;
        this.nSamples = nSamples;
        this.shardPath = shardPath;
        this.nShards = nShards;
        this.featureLocatorPath = featureLocatorPath;
        this.featureLocatorFormat = featureLocatorFormat;
        this.featureIdType = featureIdType;
        this.valuesType = valuesType;
        this.validType = validType;
        this.statsYCol = statsYCol;
    }

    public ShardManifest(
            String sampleMetaPath,
            int nSamples,
            String shardPath,
            int nShards,
            String featureLocatorPath,
            String featureLocatorFormat,
            String featureIdType,
            String valuesType,
            String validType) {
        this(
                sampleMetaPath,
                nSamples,
                shardPath,
                nShards,
                featureLocatorPath,
                featureLocatorFormat,
                featureIdType,
                valuesType,
                validType,
                null
        );
    }

    public String shardFilePath(int shardId) {
        return new File(shardPath, String.format("shard_%04d.parquet", shardId)).getPath();
    }
}
