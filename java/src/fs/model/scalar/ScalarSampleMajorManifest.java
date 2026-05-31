package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Scalar raw-sample stage를 최종 dense-long shard로 넘기기 위한 sample-major manifest입니다.
 *
 * <p>각 entry는 이미 commit이 끝난 sample parquet 파일 하나를 가리킵니다. 파일은
 * {@code (sample_id, feature_id, value)} long schema를 가져야 하며, 최종 shard build는
 * 이 목록을 읽어 dense-long part parquet로 materialize합니다.</p>
 */
public class ScalarSampleMajorManifest {
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final List<String> samplePaths;
    public final List<Long> sampleIds;
    public final String sampleIdCol;
    public final String featureIdCol;
    public final String valueCol;

    public ScalarSampleMajorManifest(
            String sampleMetaPath,
            String featureMetaPath,
            List<String> samplePaths,
            List<Long> sampleIds,
            String sampleIdCol,
            String featureIdCol,
            String valueCol) {
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.samplePaths = Collections.unmodifiableList(new ArrayList<String>(samplePaths));
        this.sampleIds = sampleIds == null
                ? null
                : Collections.unmodifiableList(new ArrayList<Long>(sampleIds));
        this.sampleIdCol = sampleIdCol;
        this.featureIdCol = featureIdCol;
        this.valueCol = valueCol;
    }
}
