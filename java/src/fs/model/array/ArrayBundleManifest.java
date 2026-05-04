package fs.model.array;

import fs.model.common.PointColumnSpec;

import java.io.File;
import java.util.List;

/**
 * Array bundle stage의 샘플 메타, feature 메타, point schema 위치를 묶어 둔 manifest다.
 */
public class ArrayBundleManifest {
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final String bundlePath;
    public final int nBundles;
    public final String featureIdType;
    public final String flagsType;
    public final List<PointColumnSpec> pointSchema;

    public ArrayBundleManifest(
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            String bundlePath,
            int nBundles,
            String featureIdType,
            String flagsType,
            List<PointColumnSpec> pointSchema) {
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.bundlePath = bundlePath;
        this.nBundles = nBundles;
        this.featureIdType = featureIdType;
        this.flagsType = flagsType;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
    }

    public String bundleFilePath(int bundleId) {
        return new File(bundlePath, String.format("bundle_%06d.parquet", bundleId)).getPath();
    }
}
