package fs.model;

public class FeatureLocation {
    public final int featureId;
    public final int globalRank;
    public final int shardId;
    public final int offsetInShard;
    public final double r2y;
    public final int nYOverlap;

    public FeatureLocation(int featureId, int globalRank, int shardId, int offsetInShard, double r2y, int nYOverlap) {
        this.featureId = featureId;
        this.globalRank = globalRank;
        this.shardId = shardId;
        this.offsetInShard = offsetInShard;
        this.r2y = r2y;
        this.nYOverlap = nYOverlap;
    }

    public FeatureLocation(int featureId, int globalRank, int shardId, int offsetInShard) {
        this(featureId, globalRank, shardId, offsetInShard, Double.NaN, -1);
    }
}
