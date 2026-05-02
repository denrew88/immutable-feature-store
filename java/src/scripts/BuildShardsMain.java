package scripts;

import fs.config.BuildShardConfig;
import fs.io.ScalarSampleBundleManifestIO;
import fs.io.ShardBuilder;
import fs.model.ScalarSampleBundleManifest;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class BuildShardsMain {
    public static void main(String[] args) throws Exception {
        String sampleMeta = getArg(args, "--sample-meta", null);
        String sampleBundleManifest = getArg(args, "--sample-bundle-manifest", null);
        String outDir = getArg(args, "--out-dir", null);
        if ((sampleMeta == null && sampleBundleManifest == null) || outDir == null) {
            System.err.println("Usage: (--sample-meta <path> | --sample-bundle-manifest <path>) --out-dir <dir> [--feature-meta <path>] [--target-shard-mb MB] [--n-shards N] [--stats-y-col COL ...]");
            System.exit(1);
        }
        if (sampleMeta != null && sampleBundleManifest != null) {
            System.err.println("Provide at most one of --sample-meta and --sample-bundle-manifest");
            System.exit(1);
        }
        BuildShardConfig cfg = new BuildShardConfig();
        String nShards = getArg(args, "--n-shards", null);
        if (nShards != null) {
            cfg.nShards = Integer.parseInt(nShards);
        }
        String targetShardMb = getArg(args, "--target-shard-mb", null);
        if (targetShardMb != null) {
            cfg.targetShardBytes = Long.parseLong(targetShardMb) * 1024L * 1024L;
        }
        String featureIdCol = getArg(args, "--feature-id-col", null);
        if (featureIdCol != null) {
            cfg.featureIdCol = featureIdCol;
        }
        String valueCol = getArg(args, "--value-col", null);
        if (valueCol != null) {
            cfg.valueCol = valueCol;
        }
        String sampleIdCol = getArg(args, "--sample-id-col", null);
        if (sampleIdCol != null) {
            cfg.sampleIdCol = sampleIdCol;
        }
        String sampleKeyCol = getArg(args, "--sample-key-col", null);
        if (sampleKeyCol != null) {
            cfg.sampleKeyCol = sampleKeyCol;
        }
        String featureKeyCol = getArg(args, "--feature-key-col", null);
        if (featureKeyCol != null) {
            cfg.featureKeyCol = featureKeyCol;
        }
        String pathCol = getArg(args, "--path-col", null);
        if (pathCol != null) {
            cfg.pathCol = pathCol;
        }
        String yCol = getArg(args, "--y-col", null);
        if (yCol != null) {
            cfg.yCol = yCol;
        }
        String featureMeta = getArg(args, "--feature-meta", null);
        if (featureMeta != null) {
            cfg.featureMetaPath = featureMeta;
        }
        List<String> statsYCols = getArgs(args, "--stats-y-col");
        if (!statsYCols.isEmpty()) {
            cfg.statsYCols = statsYCols;
        }

        if (sampleMeta != null) {
            File sampleMetaFile = new File(sampleMeta);
            if (!sampleMetaFile.exists()) {
                System.err.println("sample_meta not found: " + sampleMetaFile.getAbsolutePath());
                System.err.println("Hint: run Generate Synth first, or pass --sample-meta to an existing parquet file.");
                System.exit(1);
            }
        } else {
            File bundleManifestFile = new File(sampleBundleManifest);
            if (!bundleManifestFile.exists()) {
                System.err.println("sample-bundle manifest not found: " + bundleManifestFile.getAbsolutePath());
                System.exit(1);
            }
            ScalarSampleBundleManifest stage = ScalarSampleBundleManifestIO.read(sampleBundleManifest);
            if (cfg.featureMetaPath == null || cfg.featureMetaPath.isEmpty()) {
                cfg.featureMetaPath = stage.featureMetaPath;
            }
        }

        int exitCode = 0;
        long startedAt = System.currentTimeMillis();
        try {
            String manifestPath = (sampleBundleManifest != null)
                    ? ShardBuilder.buildShardsFromSampleBundles(sampleBundleManifest, outDir, cfg)
                    : ShardBuilder.buildShardsFromSampleMajor(sampleMeta, outDir, cfg);
            System.out.println(manifestPath);
        } catch (Exception e) {
            exitCode = 2;
            System.err.println("Failed to build shards: " + e.getMessage());
            e.printStackTrace(System.err);
        } finally {
            long elapsedMs = System.currentTimeMillis() - startedAt;
            String status = (exitCode == 0) ? "completed" : "failed";
            System.err.println("build_shards " + status + " elapsed_ms=" + elapsedMs);
        }
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    private static String getArg(String[] args, String key, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }

    private static List<String> getArgs(String[] args, String key) {
        ArrayList<String> out = new ArrayList<String>();
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                out.add(args[i + 1]);
            }
        }
        return out;
    }
}
