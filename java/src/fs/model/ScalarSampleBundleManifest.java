package fs.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
