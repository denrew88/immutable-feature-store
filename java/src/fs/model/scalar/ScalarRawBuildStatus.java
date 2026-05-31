package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * scalar raw-sample stage의 현재 진행 상태다.
 *
 * <p>raw-sample builder는 sample을 임의 순서로 받을 수 있으므로, 순차 builder처럼
 * watermark 하나만으로는 충분하지 않다. supervisor는 {@link #pendingSampleIds}를 읽어
 * 아직 완료되지 않은 sample만 worker에게 나눠주면 된다.
 */
public final class ScalarRawBuildStatus {
    public final int nSamples;
    public final int completedSampleCount;
    public final List<Long> completedSampleIds;
    public final List<Long> pendingSampleIds;
    public final Long nextPendingSampleId;
    public final boolean finishedStage;
    public final String bundleManifestPath;

    public ScalarRawBuildStatus(
            int nSamples,
            List<Long> completedSampleIds,
            List<Long> pendingSampleIds,
            boolean finishedStage,
            String bundleManifestPath) {
        this.nSamples = nSamples;
        this.completedSampleIds = Collections.unmodifiableList(new ArrayList<Long>(completedSampleIds));
        this.pendingSampleIds = Collections.unmodifiableList(new ArrayList<Long>(pendingSampleIds));
        this.completedSampleCount = completedSampleIds.size();
        this.nextPendingSampleId = pendingSampleIds.isEmpty() ? null : pendingSampleIds.get(0);
        this.finishedStage = finishedStage;
        this.bundleManifestPath = bundleManifestPath == null ? "" : bundleManifestPath;
    }
}
