package fs.model.scalar;

/**
 * dense-long scalar shard의 part parquet 하나를 설명한다.
 */
public final class ScalarDenseLongPart {
    public final int partId;
    public final String path;
    public final int firstFeatureId;
    public final int lastFeatureId;
    public final int featureCount;
    public final long rowCount;
    public final long byteSize;

    public ScalarDenseLongPart(
            int partId,
            String path,
            int firstFeatureId,
            int lastFeatureId,
            int featureCount,
            long rowCount,
            long byteSize) {
        this.partId = partId;
        this.path = path;
        this.firstFeatureId = firstFeatureId;
        this.lastFeatureId = lastFeatureId;
        this.featureCount = featureCount;
        this.rowCount = rowCount;
        this.byteSize = byteSize;
    }
}
