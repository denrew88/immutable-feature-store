package fs.io.array;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.array.ArrayBundleManifest;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Array bundle manifest JSON을 읽고 쓰는 helper다.
 */
public class ArrayBundleManifestIO {
    public static void write(ArrayBundleManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));
        root.put("n_samples", manifest.nSamples);
        root.put("bundle_path", relativeTo(path, manifest.bundlePath));
        root.put("n_bundles", manifest.nBundles);
        root.put("feature_id_dtype", manifest.featureIdType);
        root.put("flags_dtype", manifest.flagsType);

        ArrayNode schema = root.putArray("point_schema");
        for (PointColumnSpec spec : manifest.pointSchema) {
            ObjectNode item = schema.addObject();
            item.put("name", spec.name);
            item.put("storage_type", spec.storageType.value);
            item.put("logical_type", spec.logicalType.value);
            if (spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty()) {
                item.put("dictionary_path", relativeTo(path, spec.dictionaryPath));
            }
        }
        JsonUtils.writeJson(path, root);
    }

    public static ArrayBundleManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        return new ArrayBundleManifest(
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                intOrZero(root, "n_samples"),
                resolveAgainst(path, textOrEmpty(root, "bundle_path")),
                intOrZero(root, "n_bundles"),
                textOrEmpty(root, "feature_id_dtype"),
                textOrEmpty(root, "flags_dtype"),
                parsePointSchema(path, root.get("point_schema")));
    }

    private static List<PointColumnSpec> parsePointSchema(String manifestPath, JsonNode raw) {
        if (raw == null || !raw.isArray() || raw.size() == 0) {
            throw new IllegalArgumentException("array bundle manifest must include point_schema");
        }
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(raw.size());
        for (JsonNode item : raw) {
            out.add(new PointColumnSpec(
                    textOrEmpty(item, "name"),
                    StorageType.fromValue(textOrEmpty(item, "storage_type")),
                    LogicalType.fromValue(textOrEmpty(item, "logical_type")),
                    resolveAgainst(manifestPath, textOrEmpty(item, "dictionary_path"))));
        }
        return PointColumnSpec.normalizeList(out);
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

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }
}
