package fs.model.scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Scalar sample-major bundle stage의 메타데이터 위치와 sample bundle 목록을 담는 manifest다.
 */
public class ScalarSampleBundleManifest {
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final List<String> bundlePaths;
    public final String sampleIdCol;
    public final String featureIdCol;
    public final String valueCol;

    public ScalarSampleBundleManifest(
            String sampleMetaPath,
            String featureMetaPath,
            List<String> bundlePaths,
            String sampleIdCol,
            String featureIdCol,
            String valueCol) {
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.bundlePaths = Collections.unmodifiableList(new ArrayList<String>(bundlePaths));
        this.sampleIdCol = sampleIdCol;
        this.featureIdCol = featureIdCol;
        this.valueCol = valueCol;
    }
}
