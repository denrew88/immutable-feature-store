package scripts;

import fs.config.BuildShardConfig;
import fs.io.scalar.ScalarSampleMajorManifestIO;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.model.scalar.ScalarSampleMajorManifest;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Scalar sample-major stage에서 shard를 빌드하는 CLI 엔트리포인트다.
 */
public class BuildShardsMain {
    public static void main(String[] args) throws Exception {
        String sampleMajorManifest = getArg(args, "--sample-major-manifest", null);
        String outDir = getArg(args, "--out-dir", null);
        if (sampleMajorManifest == null || outDir == null) {
            System.err.println("Usage: --sample-major-manifest <path> --out-dir <dir> [--feature-meta <path>] [--target-shard-mb MB] [--stats-y-col COL ...]");
            System.exit(1);
        }
        BuildShardConfig cfg = new BuildShardConfig();
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

        File sampleMajorManifestFile = new File(sampleMajorManifest);
        if (!sampleMajorManifestFile.exists()) {
            System.err.println("sample-major manifest not found: " + sampleMajorManifestFile.getAbsolutePath());
            System.exit(1);
        }
        ScalarSampleMajorManifest stage = ScalarSampleMajorManifestIO.read(sampleMajorManifest);
        if (cfg.featureMetaPath == null || cfg.featureMetaPath.isEmpty()) {
            cfg.featureMetaPath = stage.featureMetaPath;
        }

        int exitCode = 0;
        long startedAt = System.currentTimeMillis();
        try {
            String manifestPath = ScalarDenseLongShardBuilder.buildFromSampleMajorManifest(sampleMajorManifest, outDir, cfg);
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
