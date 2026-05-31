package fs.model.array_sample_parquet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * array_sample_parquet build session의 완료/미완료 sample 현황입니다.
 *
 * <p>현재 builder는 sample id 순서를 강제하지 않습니다. supervisor는
 * {@link #pendingSampleIds}를 worker에게 나눠줄 수 있고, 순차 실행을 원하면
 * 이 목록을 앞에서부터 처리하면 됩니다.</p>
 */
public class ArraySampleParquetBuildSessionStatus {
    public final int nSamples;
    public final int completedSampleCount;
    public final int pendingSampleCount;
    public final List<Long> completedSampleIds;
    public final List<Long> pendingSampleIds;
    public final Long nextPendingSampleId;
    public final boolean finished;
    public final String manifestPath;
    public final Long inProgressSampleId;
    public final String inProgressSampleKey;

    public ArraySampleParquetBuildSessionStatus(
            int nSamples,
            List<Long> completedSampleIds,
            List<Long> pendingSampleIds,
            boolean finished,
            String manifestPath,
            Long inProgressSampleId,
            String inProgressSampleKey) {
        this.nSamples = nSamples;
        this.completedSampleIds = Collections.unmodifiableList(new ArrayList<Long>(completedSampleIds));
        this.pendingSampleIds = Collections.unmodifiableList(new ArrayList<Long>(pendingSampleIds));
        this.completedSampleCount = completedSampleIds.size();
        this.pendingSampleCount = pendingSampleIds.size();
        this.nextPendingSampleId = pendingSampleIds.isEmpty() ? null : pendingSampleIds.get(0);
        this.finished = finished;
        this.manifestPath = manifestPath == null ? "" : manifestPath;
        this.inProgressSampleId = inProgressSampleId;
        this.inProgressSampleKey = inProgressSampleKey == null ? "" : inProgressSampleKey;
    }
}
