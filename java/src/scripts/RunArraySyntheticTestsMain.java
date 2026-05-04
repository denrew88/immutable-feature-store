package scripts;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.config.ArraySyntheticConfig;
import fs.io.ArrayFeatureLocatorIndex;
import fs.io.ArrayFeatureIdIndex;
import fs.io.ArraySampleIdIndex;
import fs.io.ArrayShardManifestIO;
import fs.io.ArrayBinaryShardReader;
import fs.model.ArrayShardManifest;
import fs.model.ArraySyntheticArtifacts;
import fs.model.ArrayTrace;
import fs.synth.ArraySyntheticGenerator;

import java.io.File;
import java.util.Map;

public class RunArraySyntheticTestsMain {
    public static void main(String[] args) throws Exception {
        File root = new File("data/tmp_java_array_synth_test");
        deleteRecursively(root);
        if (!root.mkdirs()) {
            throw new IllegalStateException("failed to create test dir: " + root.getAbsolutePath());
        }

        ArraySyntheticConfig synthCfg = new ArraySyntheticConfig();
        synthCfg.nSamples = 24;
        synthCfg.nFeatures = 12;
        synthCfg.minTraceLen = 24;
        synthCfg.maxTraceLen = 64;
        synthCfg.missingFeatureRate = 0.0;
        synthCfg.emptyTraceRate = 0.0;
        synthCfg.seed = 7L;

        ArrayBundleConfig bundleCfg = new ArrayBundleConfig();
        bundleCfg.maxBundleRows = 64;
        bundleCfg.maxBundleBytes = 1L << 20;

        ArrayShardConfig shardCfg = new ArrayShardConfig();
        shardCfg.nShards = 0;
        shardCfg.targetShardBytes = 1L << 20;
        shardCfg.samplesPerBlock = 6;

        ArraySyntheticArtifacts out = ArraySyntheticGenerator.generate(
                new File(root, "bundles").getAbsolutePath(),
                new File(root, "sample_meta.parquet").getAbsolutePath(),
                synthCfg,
                bundleCfg,
                new File(root, "shards").getAbsolutePath(),
                shardCfg);

        ArrayShardManifest manifest = ArrayShardManifestIO.read(out.shardManifestPath);
        require(manifest.samplesPerBlock == 6, "samples_per_block mismatch");
        require(manifest.nShards >= 1, "expected at least one shard");
        require(manifest.nFeatures == synthCfg.nFeatures, "n_features mismatch");
        ArrayFeatureLocatorIndex locatorIndex = ArrayFeatureLocatorIndex.load(manifest);
        ArrayFeatureIdIndex featureIdIndex = ArrayFeatureIdIndex.load(out.featureMetaPath);
        ArraySampleIdIndex sampleIdIndex = ArraySampleIdIndex.load(out.sampleMetaPath);
        require(sampleIdIndex.findSampleId(5L) != null && sampleIdIndex.findSampleId(5L) == 5L, "sample id index mismatch");
        require(sampleIdIndex.findSampleIdByKey("sample_000005") != null && sampleIdIndex.findSampleIdByKey("sample_000005") == 5L, "sample key index mismatch");
        require(featureIdIndex.findFeatureIdByKey("feature_000000") != null && featureIdIndex.findFeatureIdByKey("feature_000000") == 0, "feature key index mismatch");
        require(new File(manifest.blocksIndexPath(0)).exists(), "missing blocks.idx");
        require(new File(manifest.blocksDataPath(0)).exists(), "missing blocks.bin");
        require(new File(manifest.featureMetaPath).exists(), "missing feature_meta.parquet");

        try (ArrayBinaryShardReader reader = new ArrayBinaryShardReader(manifest)) {
            Map<Long, ArrayTrace> traces = reader.loadFeatureSamplesBySampleIds(
                    0,
                    new long[]{0L, 3L, 7L, 11L},
                    locatorIndex,
                    sampleIdIndex);
            require(traces.size() == 4, "trace count mismatch");
            for (long sampleId : new long[]{0L, 3L, 7L, 11L}) {
                ArrayTrace trace = traces.get(sampleId);
                require(trace != null, "missing trace for sample_id=" + sampleId);
                require(trace.sampleId == sampleId, "sample id mismatch for sample_id=" + sampleId);
                double[] time = (double[]) trace.columns.get("time");
                double[] value = (double[]) trace.columns.get("value");
                require(time.length == value.length, "shape mismatch for sample_id=" + sampleId);
                require(trace.flags != 0, "expected present trace for sample_id=" + sampleId);
            }

            Map<String, ArrayTrace> tracesByKey = reader.loadFeatureSamplesByKeys(
                    "feature_000000",
                    new String[]{"sample_000000", "sample_000003", "sample_000007", "sample_000011"},
                    locatorIndex,
                    featureIdIndex,
                    sampleIdIndex);
            require(tracesByKey.size() == 4, "trace-by-key count mismatch");
            for (String sampleKey : new String[]{"sample_000000", "sample_000003", "sample_000007", "sample_000011"}) {
                ArrayTrace trace = tracesByKey.get(sampleKey);
                require(trace != null, "missing trace for sample_key=" + sampleKey);
                require(trace.flags != 0, "expected present trace for sample_key=" + sampleKey);
            }
        }

        System.out.println("java array synthetic tests passed");
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }
}
