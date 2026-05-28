package fs.model.array_sample_parquet;

/**
 * resume 가능한 sample-major Parquet build session의 checkpoint 상태다.
 */
public class ArraySampleParquetBuildSessionStatus {
    public final Long lastCommittedSampleId;
    public final String lastCommittedSampleKey;
    public final long nextExpectedSampleId;
    public final String nextExpectedSampleKey;
    public final int committedPartCount;
    public final boolean finished;
    public final String manifestPath;
    public final Long bufferedThroughSampleId;
    public final String bufferedThroughSampleKey;
    public final Long inProgressSampleId;
    public final String inProgressSampleKey;

    public ArraySampleParquetBuildSessionStatus(
            Long lastCommittedSampleId,
            String lastCommittedSampleKey,
            long nextExpectedSampleId,
            String nextExpectedSampleKey,
            int committedPartCount,
            boolean finished,
            String manifestPath,
            Long bufferedThroughSampleId,
            String bufferedThroughSampleKey,
            Long inProgressSampleId,
            String inProgressSampleKey) {
        this.lastCommittedSampleId = lastCommittedSampleId;
        this.lastCommittedSampleKey = lastCommittedSampleKey;
        this.nextExpectedSampleId = nextExpectedSampleId;
        this.nextExpectedSampleKey = nextExpectedSampleKey;
        this.committedPartCount = committedPartCount;
        this.finished = finished;
        this.manifestPath = manifestPath;
        this.bufferedThroughSampleId = bufferedThroughSampleId;
        this.bufferedThroughSampleKey = bufferedThroughSampleKey;
        this.inProgressSampleId = inProgressSampleId;
        this.inProgressSampleKey = inProgressSampleKey;
    }
}
