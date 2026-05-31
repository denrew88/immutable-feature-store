package fs.io.scalar;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.scalar.ScalarDenseLongManifest;
import fs.model.scalar.ScalarDenseLongPart;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * dense-long scalar shard manifest JSON을 읽고 쓰는 helper다.
 */
public final class ScalarDenseLongManifestIO {
    public static final String FORMAT = "scalar-dense-long-shard-v1";

    private ScalarDenseLongManifestIO() {
    }

    public static void write(ScalarDenseLongManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", FORMAT);
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));
        root.put("n_samples", manifest.nSamples);
        root.put("n_features", manifest.nFeatures);
        root.put("parts_path", relativeTo(path, manifest.partsPath));

        ArrayNode parts = root.putArray("parts");
        for (ScalarDenseLongPart part : manifest.parts) {
            ObjectNode item = parts.addObject();
            item.put("part_id", part.partId);
            item.put("path", relativeTo(path, part.path));
            item.put("first_feature_id", part.firstFeatureId);
            item.put("last_feature_id", part.lastFeatureId);
            item.put("feature_count", part.featureCount);
            item.put("row_count", part.rowCount);
            item.put("byte_size", part.byteSize);
        }

        root.put("feature_locator_path", relativeTo(path, manifest.featureLocatorPath));
        root.put("id_scheme", "dense_row_ids");
        root.put("sample_key_col", manifest.sampleKeyCol);
        root.put("feature_key_col", manifest.featureKeyCol);
        root.put("sample_id_col", manifest.sampleIdCol);
        root.put("feature_id_col", manifest.featureIdCol);
        root.put("value_col", manifest.valueCol);
        root.put("mask_col", manifest.maskCol);
        root.put("compression", manifest.compression);
        root.put("row_group_features", manifest.rowGroupFeatures);
        if (manifest.targetPartBytes != null) {
            root.put("target_part_bytes", manifest.targetPartBytes.longValue());
        }
        if (!manifest.selectionStats.isEmpty()) {
            ObjectNode stats = root.putObject("selection_stats");
            for (Map.Entry<String, String> entry : manifest.selectionStats.entrySet()) {
                stats.put(entry.getKey(), relativeTo(path, entry.getValue()));
            }
        }
        JsonUtils.writeJsonAtomic(path, root);
    }

    public static ScalarDenseLongManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        String format = textOrEmpty(root, "format");
        if (!FORMAT.equals(format)) {
            throw new IOException("unsupported dense-long manifest format: " + format);
        }
        ArrayList<ScalarDenseLongPart> parts = new ArrayList<ScalarDenseLongPart>();
        JsonNode rawParts = root.get("parts");
        if (rawParts != null && rawParts.isArray()) {
            for (JsonNode item : rawParts) {
                parts.add(new ScalarDenseLongPart(
                        item.path("part_id").asInt(),
                        resolveAgainst(path, item.path("path").asText()),
                        item.path("first_feature_id").asInt(),
                        item.path("last_feature_id").asInt(),
                        item.path("feature_count").asInt(),
                        item.path("row_count").asLong(),
                        item.path("byte_size").asLong()));
            }
        }
        return new ScalarDenseLongManifest(
                new File(path).getAbsolutePath(),
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                root.path("n_samples").asInt(),
                root.path("n_features").asInt(),
                resolveAgainst(path, textOrEmpty(root, "parts_path")),
                parts,
                resolveAgainst(path, textOrEmpty(root, "feature_locator_path")),
                defaultText(root, "sample_key_col", "sample_key"),
                defaultText(root, "feature_key_col", "feature_key"),
                defaultText(root, "sample_id_col", "sample_id"),
                defaultText(root, "feature_id_col", "feature_id"),
                defaultText(root, "value_col", "value"),
                defaultText(root, "mask_col", "mask"),
                defaultText(root, "compression", "zstd"),
                root.path("row_group_features").asInt(128),
                root.has("target_part_bytes") && !root.get("target_part_bytes").isNull()
                        ? Long.valueOf(root.get("target_part_bytes").asLong())
                        : null,
                parseSelectionStats(path, root.get("selection_stats")));
    }

    private static Map<String, String> parseSelectionStats(String manifestPath, JsonNode raw) {
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        if (raw == null || raw.isNull() || !raw.isObject()) {
            return out;
        }
        java.util.Iterator<String> names = raw.fieldNames();
        while (names.hasNext()) {
            String name = names.next();
            out.put(name, resolveAgainst(manifestPath, raw.get(name).asText()));
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

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return child == null || child.isNull() ? "" : child.asText();
    }

    private static String defaultText(JsonNode node, String fieldName, String fallback) {
        String value = textOrEmpty(node, fieldName);
        return value.isEmpty() ? fallback : value;
    }
}
