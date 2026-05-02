package scripts;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.config.ArraySyntheticConfig;
import fs.model.ArraySyntheticArtifacts;
import fs.synth.ArraySyntheticGenerator;

public class GenerateArraySynthMain {
    public static void main(String[] args) throws Exception {
        String bundleOutDir = getArg(args, "--bundle-out-dir", null);
        String sampleMeta = getArg(args, "--sample-meta", null);
        if (bundleOutDir == null || sampleMeta == null) {
            System.err.println("Usage: --bundle-out-dir <dir> --sample-meta <path> [--shard-out-dir <dir>]");
            System.exit(1);
        }

        ArraySyntheticConfig synthCfg = new ArraySyntheticConfig();
        String nSamples = getArg(args, "--n-samples", null);
        if (nSamples != null) {
            synthCfg.nSamples = Integer.parseInt(nSamples);
        }
        String nFeatures = getArg(args, "--n-features", null);
        if (nFeatures != null) {
            synthCfg.nFeatures = Integer.parseInt(nFeatures);
        }
        String minTraceLen = getArg(args, "--min-trace-len", null);
        if (minTraceLen != null) {
            synthCfg.minTraceLen = Integer.parseInt(minTraceLen);
        }
        String maxTraceLen = getArg(args, "--max-trace-len", null);
        if (maxTraceLen != null) {
            synthCfg.maxTraceLen = Integer.parseInt(maxTraceLen);
        }
        String seed = getArg(args, "--seed", null);
        if (seed != null) {
            synthCfg.seed = Long.parseLong(seed);
        }

        ArrayBundleConfig bundleCfg = new ArrayBundleConfig();
        String maxBundleRows = getArg(args, "--max-bundle-rows", null);
        if (maxBundleRows != null) {
            bundleCfg.maxBundleRows = Integer.parseInt(maxBundleRows);
        }
        String maxBundleBytes = getArg(args, "--max-bundle-bytes", null);
        if (maxBundleBytes != null) {
            bundleCfg.maxBundleBytes = Long.parseLong(maxBundleBytes);
        }

        ArrayShardConfig shardCfg = new ArrayShardConfig();
        String targetShardMb = getArg(args, "--target-shard-mb", null);
        if (targetShardMb != null) {
            shardCfg.targetShardBytes = Long.parseLong(targetShardMb) * 1024L * 1024L;
        }
        String nShards = getArg(args, "--n-shards", null);
        if (nShards != null) {
            shardCfg.nShards = Integer.parseInt(nShards);
        }
        String samplesPerBlock = getArg(args, "--samples-per-block", null);
        if (samplesPerBlock != null) {
            shardCfg.samplesPerBlock = Integer.parseInt(samplesPerBlock);
        }

        String shardOutDir = getArg(args, "--shard-out-dir", null);
        ArraySyntheticArtifacts out = ArraySyntheticGenerator.generate(
                bundleOutDir,
                sampleMeta,
                synthCfg,
                bundleCfg,
                shardOutDir,
                shardCfg);
        System.out.println((out.shardManifestPath != null) ? out.shardManifestPath : out.bundleManifestPath);
    }

    private static String getArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
