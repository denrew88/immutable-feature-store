package scripts;

import fs.config.BuildShardConfig;
import fs.io.ShardBuilder;
import java.io.File;

public class BuildShardsMain {
    public static void main(String[] args) throws Exception {
        String sampleMeta = getArg(args, "--sample-meta", null);
        String outDir = getArg(args, "--out-dir", null);
        if (sampleMeta == null || outDir == null) {
            System.err.println("Usage: --sample-meta <path> --out-dir <dir> [--n-shards N]");
            System.exit(1);
        }
        File sampleMetaFile = new File(sampleMeta);
        if (!sampleMetaFile.exists()) {
            System.err.println("sample_meta not found: " + sampleMetaFile.getAbsolutePath());
            System.err.println("Hint: run Generate Synth first, or pass --sample-meta to an existing parquet file.");
            System.exit(1);
        }
        BuildShardConfig cfg = new BuildShardConfig();
        String nShards = getArg(args, "--n-shards", null);
        if (nShards != null) {
            cfg.nShards = Integer.parseInt(nShards);
        }
        if (cfg.nShards <= 0) {
            System.err.println("--n-shards must be > 0");
            System.exit(1);
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
        String pathCol = getArg(args, "--path-col", null);
        if (pathCol != null) {
            cfg.pathCol = pathCol;
        }
        String yCol = getArg(args, "--y-col", null);
        if (yCol != null) {
            cfg.yCol = yCol;
        }

        int exitCode = 0;
        long startedAt = System.currentTimeMillis();
        try {
            String manifestPath = ShardBuilder.buildShardsFromSampleMajor(sampleMeta, outDir, cfg);
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
}
