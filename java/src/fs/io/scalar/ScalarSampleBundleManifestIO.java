package fs.io.scalar;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.scalar.ScalarSampleBundleManifest;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Scalar sample-major bundle manifest JSON을 읽고 쓰는 helper다.
 */
public final class ScalarSampleBundleManifestIO {
    private ScalarSampleBundleManifestIO() {
    }

    public static void write(ScalarSampleBundleManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", "scalar-sample-bundles");
        root.put("version", 1);
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));

        ArrayNode bundlePaths = root.putArray("bundle_paths");
        for (String bundlePath : manifest.bundlePaths) {
            bundlePaths.add(relativeTo(path, bundlePath));
        }
        root.put("sample_id_col", manifest.sampleIdCol);
        root.put("feature_id_col", manifest.featureIdCol);
        root.put("value_col", manifest.valueCol);
        JsonUtils.writeJson(path, root);
    }

    public static ScalarSampleBundleManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        String format = textOrEmpty(root, "format");
        if (!"scalar-sample-bundles".equals(format)) {
            throw new IOException("unsupported sample-bundle manifest format: " + format);
        }
        JsonNode rawBundlePaths = root.get("bundle_paths");
        ArrayList<String> bundlePaths = new ArrayList<String>();
        if (rawBundlePaths != null && rawBundlePaths.isArray()) {
            for (JsonNode item : rawBundlePaths) {
                bundlePaths.add(resolveAgainst(path, item.asText()));
            }
        }
        return new ScalarSampleBundleManifest(
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                bundlePaths,
                defaultText(root, "sample_id_col", "sample_id"),
                defaultText(root, "feature_id_col", "feature_id"),
                defaultText(root, "value_col", "value"));
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
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static String defaultText(JsonNode node, String fieldName, String fallback) {
        String out = textOrEmpty(node, fieldName);
        return out.isEmpty() ? fallback : out;
    }
}
