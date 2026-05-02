package fs.model;

public class Candidate {
    public final int featureId;
    public final int shardId;
    public final int offsetInShard;
    public final double r2y;
    public final int nValidY;

    public Candidate(int featureId, int shardId, int offsetInShard, double r2y, int nValidY) {
        this.featureId = featureId;
        this.shardId = shardId;
        this.offsetInShard = offsetInShard;
        this.r2y = r2y;
        this.nValidY = nValidY;
    }
}
