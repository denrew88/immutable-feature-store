package scripts;

import fs.config.BuildShardConfig;
import fs.config.SelectionConfig;
import fs.io.DuckDBShardReader;
import fs.io.ManifestIO;
import fs.model.Candidate;
import fs.model.ShardManifest;
import fs.pipeline.CandidateBuilder;
import fs.pipeline.Selector;

import java.util.List;

public class RunSelectionMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", null);
        if (manifestPath == null) {
            System.err.println("Usage: --manifest <path> [--top-m M] [--y-r2 R] [--min-non-null-y N] [--ff-r2 R] [--min-non-null-pair N]");
            System.exit(1);
        }
        ShardManifest manifest = ManifestIO.read(manifestPath);

        BuildShardConfig metaCfg = new BuildShardConfig();
        String sampleIdCol = getArg(args, "--sample-id-col", null);
        if (sampleIdCol != null) {
            metaCfg.sampleIdCol = sampleIdCol;
        }
        String yCol = getArg(args, "--y-col", null);
        if (yCol != null) {
            metaCfg.yCol = yCol;
        }
        String pathCol = getArg(args, "--path-col", null);
        if (pathCol != null) {
            metaCfg.pathCol = pathCol;
        }

        SelectionConfig cfg = new SelectionConfig();
        String yR2 = getArg(args, "--y-r2", null);
        if (yR2 != null) {
            cfg.yR2Threshold = Double.parseDouble(yR2);
        }
        String minY = getArg(args, "--min-non-null-y", null);
        if (minY != null) {
            cfg.minNonNullY = Integer.parseInt(minY);
        }
        String ffR2 = getArg(args, "--ff-r2", null);
        if (ffR2 != null) {
            cfg.ffR2Threshold = Double.parseDouble(ffR2);
        }
        String minPair = getArg(args, "--min-non-null-pair", null);
        if (minPair != null) {
            cfg.minNonNullPair = Integer.parseInt(minPair);
        }
        String topM = getArg(args, "--top-m", null);
        if (topM != null) {
            cfg.topM = Integer.parseInt(topM);
        }
        String initialCap = getArg(args, "--initial-cap", null);
        if (initialCap != null) {
            cfg.initialCap = Integer.parseInt(initialCap);
        }
        String maxStep = getArg(args, "--max-step", null);
        if (maxStep != null) {
            cfg.maxStep = Integer.parseInt(maxStep);
        }
        String batchSize = getArg(args, "--batch-size", null);
        if (batchSize != null) {
            cfg.batchSize = Integer.parseInt(batchSize);
        }
        String maxGap = getArg(args, "--max-gap", null);
        if (maxGap != null) {
            cfg.maxGap = Integer.parseInt(maxGap);
        }
        String maxCandidates = getArg(args, "--max-candidates", null);
        if (maxCandidates != null) {
            cfg.maxCandidates = Integer.parseInt(maxCandidates);
        }

        List<Candidate> candidates = CandidateBuilder.buildCandidatesFromShards(manifest, cfg, metaCfg.yCol, metaCfg);
        try (DuckDBShardReader reader = new DuckDBShardReader(manifest, cfg.maxGap)) {
            List<Candidate> selected = Selector.selectFeaturesIncremental(candidates, reader, cfg);
            for (Candidate c : selected) {
                System.out.println(c.featureId);
            }
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
