package scripts;

import fs.io.scalar.FeatureLocatorIndex;
import fs.io.scalar.ManifestIO;
import fs.model.common.FeatureLocation;
import fs.model.scalar.ShardManifest;

/**
 * Scalar feature가 어느 shard row에 저장돼 있는지 출력하는 디버그용 CLI다.
 */
public class LocateFeatureMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", null);
        String featureIdArg = getArg(args, "--feature-id", null);
        if (manifestPath == null || featureIdArg == null) {
            System.err.println("Usage: --manifest <path> --feature-id <signed_int32>");
            System.exit(1);
        }
        int featureId = Integer.parseInt(featureIdArg);
        ShardManifest manifest = ManifestIO.read(manifestPath);
        try (FeatureLocatorIndex idx = FeatureLocatorIndex.load(manifest)) {
            FeatureLocation loc = idx.find(featureId);
            if (loc == null) {
                System.out.println("NOT_FOUND\tfeature_id=" + featureId);
                return;
            }
            String shardPath = manifest.shardFilePath(loc.shardId);
            System.out.println("feature_id=" + loc.featureId
                    + "\tglobal_rank=" + loc.globalRank
                    + "\tshard_id=" + loc.shardId
                    + "\toffset_in_shard=" + loc.offsetInShard
                    + "\tr2y=" + loc.r2y
                    + "\tn_y_overlap=" + loc.nYOverlap
                    + "\tshard_path=" + shardPath);
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
