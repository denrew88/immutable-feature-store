package fs.model.array;

/**
 * array resumable build session의 현재 체크포인트 상태를 나타낸다.
 *
 * <p>재시작 시에는 {@code lastCommittedSampleId}까지가 안전하게 seal된 bundle에
 * 들어간 상태라고 보면 되고, 사용자는 보통 {@code nextExpectedSampleId}부터 다시
 * 데이터를 준비해서 넣으면 된다.
 */
public final class ArrayBuildSessionStatus {
    public final Long lastCommittedSampleId;
    public final String lastCommittedSampleKey;
    public final long nextExpectedSampleId;
    public final String nextExpectedSampleKey;
    public final int committedBundleCount;
    public final boolean finishedStage;
    public final String bundleManifestPath;
    public final Long bufferedThroughSampleId;
    public final String bufferedThroughSampleKey;
    public final Long inProgressSampleId;
    public final String inProgressSampleKey;

    public ArrayBuildSessionStatus(
            Long lastCommittedSampleId,
            String lastCommittedSampleKey,
            long nextExpectedSampleId,
            String nextExpectedSampleKey,
            int committedBundleCount,
            boolean finishedStage,
            String bundleManifestPath,
            Long bufferedThroughSampleId,
            String bufferedThroughSampleKey,
            Long inProgressSampleId,
            String inProgressSampleKey) {
        this.lastCommittedSampleId = lastCommittedSampleId;
        this.lastCommittedSampleKey = lastCommittedSampleKey;
        this.nextExpectedSampleId = nextExpectedSampleId;
        this.nextExpectedSampleKey = nextExpectedSampleKey;
        this.committedBundleCount = committedBundleCount;
        this.finishedStage = finishedStage;
        this.bundleManifestPath = (bundleManifestPath == null) ? "" : bundleManifestPath;
        this.bufferedThroughSampleId = bufferedThroughSampleId;
        this.bufferedThroughSampleKey = bufferedThroughSampleKey;
        this.inProgressSampleId = inProgressSampleId;
        this.inProgressSampleKey = inProgressSampleKey;
    }
}
