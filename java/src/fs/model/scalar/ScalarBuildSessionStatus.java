package fs.model.scalar;

/**
 * scalar resumable build session의 현재 체크포인트 상태를 나타낸다.
 *
 * <p>사용자는 보통 {@code lastCommittedSampleId}와
 * {@code nextExpectedSampleId}만 보면 충분하다.
 */
public final class ScalarBuildSessionStatus {
    public final Long lastCommittedSampleId;
    public final String lastCommittedSampleKey;
    public final long nextExpectedSampleId;
    public final String nextExpectedSampleKey;
    public final int committedBundleCount;
    public final boolean finishedStage;
    public final String bundleManifestPath;
    public final Long bufferedThroughSampleId;
    public final String bufferedThroughSampleKey;

    public ScalarBuildSessionStatus(
            Long lastCommittedSampleId,
            String lastCommittedSampleKey,
            long nextExpectedSampleId,
            String nextExpectedSampleKey,
            int committedBundleCount,
            boolean finishedStage,
            String bundleManifestPath,
            Long bufferedThroughSampleId,
            String bufferedThroughSampleKey) {
        this.lastCommittedSampleId = lastCommittedSampleId;
        this.lastCommittedSampleKey = lastCommittedSampleKey;
        this.nextExpectedSampleId = nextExpectedSampleId;
        this.nextExpectedSampleKey = nextExpectedSampleKey;
        this.committedBundleCount = committedBundleCount;
        this.finishedStage = finishedStage;
        this.bundleManifestPath = (bundleManifestPath == null) ? "" : bundleManifestPath;
        this.bufferedThroughSampleId = bufferedThroughSampleId;
        this.bufferedThroughSampleKey = bufferedThroughSampleKey;
    }
}
