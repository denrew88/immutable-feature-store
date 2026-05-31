package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * scalar build session의 현재 진행 상태입니다.
 *
 * <p>표준 scalar builder는 sample 하나를 raw parquet 파일 하나로 commit합니다.
 * 그래서 재개 기준은 순차 watermark가 아니라 완료된 sample id 목록과 아직 남은
 * sample id 목록입니다. supervisor는 {@link #pendingSampleIds}를 worker에게
 * 나눠줄 수 있고, 순차 실행을 원하면 이 목록을 앞에서부터 처리하면 됩니다.</p>
 */
public final class ScalarBuildSessionStatus {
    public final int nSamples;
    public final int completedSampleCount;
    public final int pendingSampleCount;
    public final List<Long> completedSampleIds;
    public final List<Long> pendingSampleIds;
    public final Long nextPendingSampleId;
    public final boolean finishedStage;
    public final String sampleMajorManifestPath;

    public ScalarBuildSessionStatus(
            int nSamples,
            List<Long> completedSampleIds,
            List<Long> pendingSampleIds,
            boolean finishedStage,
            String sampleMajorManifestPath) {
        this.nSamples = nSamples;
        this.completedSampleIds = Collections.unmodifiableList(new ArrayList<Long>(completedSampleIds));
        this.pendingSampleIds = Collections.unmodifiableList(new ArrayList<Long>(pendingSampleIds));
        this.completedSampleCount = completedSampleIds.size();
        this.pendingSampleCount = pendingSampleIds.size();
        this.nextPendingSampleId = pendingSampleIds.isEmpty() ? null : pendingSampleIds.get(0);
        this.finishedStage = finishedStage;
        this.sampleMajorManifestPath = sampleMajorManifestPath == null ? "" : sampleMajorManifestPath;
    }
}
