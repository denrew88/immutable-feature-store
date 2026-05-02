package fs.io;

import fs.config.ArrayShardConfig;
import fs.model.ArrayShardManifest;

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
}
