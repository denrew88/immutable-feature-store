package fs.io;

import fs.config.ArrayShardConfig;
import fs.model.ArrayShardManifest;
import fs.model.PointColumnSpec;

import java.util.List;
import java.util.Map;

public final class ArrayBinaryShards {
    private ArrayBinaryShards() {
    }

    public static String buildFromBundles(String bundleManifestPath, String outDir, ArrayShardConfig config) throws Exception {
        return ArrayShardBuilder.buildFromBundles(bundleManifestPath, outDir, config);
    }

    public static ArrayShardManifest loadManifest(String manifestPath) throws Exception {
        return ArrayShardManifestIO.read(manifestPath);
    }

    public static ArrayShardReader open(String manifestPath) throws Exception {
        return new ArrayShardReader(loadManifest(manifestPath));
    }

    public static ArrayFeatureLocatorIndex loadLocator(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureLocatorIndex.load(manifest);
    }

    public static ArraySampleIdIndex loadSampleIds(ArrayShardManifest manifest) throws Exception {
        return ArraySampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
    }

    public static ArrayFeatureIdIndex loadFeatureIds(ArrayShardManifest manifest) throws Exception {
        return ArrayFeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
    }

    public static String writeSampleMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeSampleMeta(records, path);
    }

    public static String writeFeatureMeta(List<Map<String, Object>> records, String path) throws Exception {
        return ArrayMetadataWriter.writeFeatureMeta(records, path);
    }

    public static ArrayDatasetBuilder newBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema);
    }
}
