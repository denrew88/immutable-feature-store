package fs.model.array;

/**
 * 특정 array feature block이 어느 shard의 몇 번째 record에 있는지 나타낸다.
 */
public class ArrayBlockLocation {
    public final int featureId;
    public final int blockId;
    public final int shardId;
    public final int rowInShard;
    public final long sampleIdStart;
    public final long sampleIdEnd;

    public ArrayBlockLocation(
            int featureId,
            int blockId,
            int shardId,
            int rowInShard,
            long sampleIdStart,
            long sampleIdEnd) {
        this.featureId = featureId;
        this.blockId = blockId;
        this.shardId = shardId;
        this.rowInShard = rowInShard;
        this.sampleIdStart = sampleIdStart;
        this.sampleIdEnd = sampleIdEnd;
    }

    public boolean containsSampleId(long sampleId) {
        return sampleId >= sampleIdStart && sampleId <= sampleIdEnd;
    }
}
