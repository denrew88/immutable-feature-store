package scripts;

import fs.config.ArrayShardConfig;
import fs.io.ArrayShardBuilder;

import java.io.File;

public class BuildArrayShardsMain {
    public static void main(String[] args) throws Exception {
        String bundleManifest = getArg(args, "--bundle-manifest", null);
        String outDir = getArg(args, "--out-dir", null);
        if (bundleManifest == null || outDir == null) {
            System.err.println("Usage: --bundle-manifest <path> --out-dir <dir> "
                    + "[--target-shard-mb N] [--n-shards N] [--samples-per-block N]");
            System.exit(1);
        }
        File manifestFile = new File(bundleManifest);
        if (!manifestFile.exists()) {
            System.err.println("bundle manifest not found: " + manifestFile.getAbsolutePath());
            System.exit(1);
        }

        ArrayShardConfig cfg = new ArrayShardConfig();
        String targetShardMb = getArg(args, "--target-shard-mb", null);
        if (targetShardMb != null) {
            cfg.targetShardBytes = Long.parseLong(targetShardMb) * 1024L * 1024L;
        }
        String nShards = getArg(args, "--n-shards", null);
        if (nShards != null) {
            cfg.nShards = Integer.parseInt(nShards);
        }
        String samplesPerBlock = getArg(args, "--samples-per-block", null);
        if (samplesPerBlock != null) {
            cfg.samplesPerBlock = Integer.parseInt(samplesPerBlock);
        }

        int exitCode = 0;
        long startedAt = System.currentTimeMillis();
        try {
            String manifestPath = ArrayShardBuilder.buildFromBundles(bundleManifest, outDir, cfg);
            System.out.println(manifestPath);
        } catch (Exception e) {
            exitCode = 2;
            System.err.println("Failed to build array shards: " + e.getMessage());
            e.printStackTrace(System.err);
        } finally {
            long elapsedMs = System.currentTimeMillis() - startedAt;
            String status = (exitCode == 0) ? "completed" : "failed";
            System.err.println("build_array_shards " + status + " elapsed_ms=" + elapsedMs);
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
