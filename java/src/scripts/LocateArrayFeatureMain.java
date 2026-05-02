package scripts;

import fs.io.ArrayFeatureLocatorIndex;
import fs.io.ArrayShardManifestIO;
import fs.model.ArrayBlockLocation;
import fs.model.ArrayShardManifest;

import java.util.List;

public class LocateArrayFeatureMain {
    public static void main(String[] args) throws Exception {
        String manifestPath = getArg(args, "--manifest", null);
        String featureIdArg = getArg(args, "--feature-id", null);
        if (manifestPath == null || featureIdArg == null) {
            System.err.println("Usage: --manifest <path> --feature-id <signed_int32>");
            System.exit(1);
        }

        int featureId = Integer.parseInt(featureIdArg);
        ArrayShardManifest manifest = ArrayShardManifestIO.read(manifestPath);
        ArrayFeatureLocatorIndex idx = ArrayFeatureLocatorIndex.load(manifest);
        List<ArrayBlockLocation> blocks = idx.blocksForFeature(featureId);
        if (blocks.isEmpty()) {
            System.out.println("NOT_FOUND\tfeature_id=" + featureId);
            return;
        }
        for (ArrayBlockLocation loc : blocks) {
            String dataPath = manifest.blocksDataPath(loc.shardId);
            System.out.println(
                    "feature_id=" + loc.featureId
                            + "\tblock_id=" + loc.blockId
                            + "\tshard_id=" + loc.shardId
                            + "\trow_in_shard=" + loc.rowInShard
                            + "\tsample_id_start=" + loc.sampleIdStart
                            + "\tsample_id_end=" + loc.sampleIdEnd
                            + "\tblocks_data_path=" + dataPath);
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
