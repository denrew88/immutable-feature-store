package fs.io.scalar;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.scalar.ShardManifest;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Scalar shard manifest JSON을 읽고 쓰는 helper다.
 */
public class ManifestIO {
    public static void write(ShardManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));
        root.put("n_samples", manifest.nSamples);
        root.put("n_features", manifest.nFeatures);
        root.put("shard_path", relativeTo(path, manifest.shardPath));
        root.put("n_shards", manifest.nShards);
        root.put("feature_locator_path", relativeTo(path, manifest.featureLocatorPath));
        root.put("feature_locator_format", manifest.featureLocatorFormat);
        root.put("feature_id_dtype", manifest.featureIdType);
        root.put("values_dtype", manifest.valuesType);
        root.put("valid_dtype", manifest.validType);
        root.put("id_scheme", manifest.idScheme);
        root.put("sample_key_col", manifest.sampleKeyCol);
        root.put("feature_key_col", manifest.featureKeyCol);

        if (manifest.targetShardBytes != null) {
            root.put("target_shard_bytes", manifest.targetShardBytes.longValue());
        }
        if (!manifest.selectionStats.isEmpty()) {
            ObjectNode selectionStats = root.putObject("selection_stats");
            for (Map.Entry<String, String> entry : manifest.selectionStats.entrySet()) {
                selectionStats.put(entry.getKey(), relativeTo(path, entry.getValue()));
            }
        } else if (manifest.statsYCol != null) {
            root.put("stats_y_col", manifest.statsYCol);
        }
        JsonUtils.writeJson(path, root);
    }

    public static ShardManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        return new ShardManifest(
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                intOrZero(root, "n_samples"),
                intOrZero(root, "n_features"),
                resolveAgainst(path, textOrEmpty(root, "shard_path")),
                intOrZero(root, "n_shards"),
                resolveAgainst(path, textOrEmpty(root, "feature_locator_path")),
                textOrEmpty(root, "feature_locator_format"),
                textOrEmpty(root, "feature_id_dtype"),
                textOrEmpty(root, "values_dtype"),
                textOrEmpty(root, "valid_dtype"),
                defaultText(root, "id_scheme", "legacy"),
                defaultText(root, "sample_key_col", "sample_key"),
                defaultText(root, "feature_key_col", "feature_key"),
                longOrNull(root, "target_shard_bytes"),
                parseSelectionStats(path, root.get("selection_stats")),
                textOrNull(root, "stats_y_col"));
    }

    private static Map<String, String> parseSelectionStats(String manifestPath, JsonNode raw) {
        if (raw == null || raw.isNull() || !raw.isObject() || raw.size() == 0) {
            return Collections.emptyMap();
        }
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        java.util.Iterator<String> fieldNames = raw.fieldNames();
        while (fieldNames.hasNext()) {
            String key = fieldNames.next();
            out.put(key, resolveAgainst(manifestPath, raw.get(key).asText()));
        }
        return out;
    }

    private static String relativeTo(String manifestPath, String targetPath) {
        if (targetPath == null || targetPath.isEmpty()) {
            return "";
        }
        File target = new File(targetPath);
        if (!target.isAbsolute()) {
            return targetPath.replace("\\", "/");
        }
        File manifestDir = new File(manifestPath).getAbsoluteFile().getParentFile();
        return manifestDir.toPath().relativize(target.getAbsoluteFile().toPath()).toString().replace("\\", "/");
    }

    private static String resolveAgainst(String manifestPath, String storedPath) {
        if (storedPath == null || storedPath.isEmpty()) {
            return "";
        }
        File file = new File(storedPath);
        if (file.isAbsolute()) {
            return file.getAbsolutePath();
        }
        File manifestDir = new File(manifestPath).getAbsoluteFile().getParentFile();
        return new File(manifestDir, storedPath).getAbsolutePath();
    }

    private static int intOrZero(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? 0 : child.asInt();
    }

    private static Long longOrNull(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? null : Long.valueOf(child.asLong());
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static String textOrNull(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? null : child.asText();
    }

    private static String defaultText(JsonNode node, String fieldName, String fallback) {
        String out = textOrNull(node, fieldName);
        return (out == null || out.isEmpty()) ? fallback : out;
    }
}
