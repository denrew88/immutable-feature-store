package scripts;

import fs.io.ScalarDenseLongDataset;
import fs.model.selection.Candidate;

import java.util.List;

/**
 * Dense-long scalar shard의 precomputed selection stats를 조회하는 CLI다.
 */
public class RunSelectionMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", null);
        if (manifestPath == null) {
            System.err.println("Usage: --manifest <path> [--top-m M] [--y-col COL]");
            System.exit(1);
        }
        String yCol = getArg(args, "--y-col", "y");
        int topM = Integer.parseInt(getArg(args, "--top-m", "100"));
        try (ScalarDenseLongDataset dataset = new ScalarDenseLongDataset(manifestPath)) {
            List<Candidate> candidates = dataset.topFeaturesFromStats(yCol, topM);
            for (Candidate c : candidates) {
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
