package fs.model;

public class ArrayBinaryShardInfo {
    public final int shardId;
    public final int featureIdStart;
    public final int featureIdEnd;
    public final int featureCount;
    public final int blockCount;
    public final String blocksIndexName;
    public final String blocksDataName;

    public ArrayBinaryShardInfo(
            int shardId,
            int featureIdStart,
            int featureIdEnd,
            int featureCount,
            int blockCount,
            String blocksIndexName,
            String blocksDataName) {
        this.shardId = shardId;
        this.featureIdStart = featureIdStart;
        this.featureIdEnd = featureIdEnd;
        this.featureCount = featureCount;
        this.blockCount = blockCount;
        this.blocksIndexName = blocksIndexName;
        this.blocksDataName = blocksDataName;
    }
}
