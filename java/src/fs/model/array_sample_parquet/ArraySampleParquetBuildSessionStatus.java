package fs.model.array_sample_parquet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * array_sample_parquet raw build session의 checkpoint 상태.
 *
 * <p>새 raw-stage 방식에서는 sample_id 순서대로만 이어 쓸 필요가 없다. supervisor는
 * {@link #pendingSampleIds}를 읽어 남은 sample들을 worker에게 나눠주면 된다.
 * {@link #nextExpectedSampleId}는 기존 순차 예제와의 호환을 위해 가장 작은 pending sample id를
 * 제공한다.</p>
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
    public final int nSamples;
    public final int completedSampleCount;
    public final int pendingSampleCount;
    public final List<Long> completedSampleIds;
    public final List<Long> pendingSampleIds;

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
        this(
                lastCommittedSampleId,
                lastCommittedSampleKey,
                nextExpectedSampleId,
                nextExpectedSampleKey,
                committedPartCount,
                finished,
                manifestPath,
                bufferedThroughSampleId,
                bufferedThroughSampleKey,
                inProgressSampleId,
                inProgressSampleKey,
                0,
                committedPartCount,
                0,
                Collections.<Long>emptyList(),
                Collections.<Long>emptyList());
    }

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
            String inProgressSampleKey,
            int nSamples,
            int completedSampleCount,
            int pendingSampleCount,
            List<Long> completedSampleIds,
            List<Long> pendingSampleIds) {
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
        this.nSamples = nSamples;
        this.completedSampleCount = completedSampleCount;
        this.pendingSampleCount = pendingSampleCount;
        this.completedSampleIds = Collections.unmodifiableList(new ArrayList<Long>(completedSampleIds));
        this.pendingSampleIds = Collections.unmodifiableList(new ArrayList<Long>(pendingSampleIds));
    }
}
