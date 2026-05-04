package fs.model.selection;

/**
 * Selection 후보 feature 하나의 점수와 샘플 수를 나타내는 모델이다.
 */
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
