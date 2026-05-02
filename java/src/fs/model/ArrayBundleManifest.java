package fs.model;

import java.io.File;

public class ArrayBundleManifest {
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final String bundlePath;
    public final int nBundles;
    public final String featureIdType;
    public final String flagsType;
    public final String timeType;
    public final String valueType;

    public ArrayBundleManifest(
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            String bundlePath,
            int nBundles,
            String featureIdType,
            String flagsType,
            String timeType,
            String valueType) {
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.bundlePath = bundlePath;
        this.nBundles = nBundles;
        this.featureIdType = featureIdType;
        this.flagsType = flagsType;
        this.timeType = timeType;
        this.valueType = valueType;
    }

    public String bundleFilePath(int bundleId) {
        return new File(bundlePath, String.format("bundle_%06d.parquet", bundleId)).getPath();
    }
}
