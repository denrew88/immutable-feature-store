package fs.io.scalar;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.JsonUtils;
import fs.model.scalar.ScalarSampleMajorManifest;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Scalar sample-major manifest JSON을 읽고 씁니다.
 *
 * <p>manifest는 raw sample parquet 파일 목록과 metadata 경로만 기록합니다. 실제 value row는
 * 각 sample parquet에 있고, dense-long shard builder가 이 manifest를 entrypoint로 사용합니다.</p>
 */
public final class ScalarSampleMajorManifestIO {
    public static final String FORMAT = "scalar-sample-major-v1";

    private ScalarSampleMajorManifestIO() {
    }

    public static void write(ScalarSampleMajorManifest manifest, String path) throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", FORMAT);
        root.put("version", 1);
        root.put("sample_meta_path", relativeTo(path, manifest.sampleMetaPath));
        root.put("feature_meta_path", relativeTo(path, manifest.featureMetaPath));

        ArrayNode samplePaths = root.putArray("sample_paths");
        for (String samplePath : manifest.samplePaths) {
            samplePaths.add(relativeTo(path, samplePath));
        }
        if (manifest.sampleIds != null) {
            ArrayNode sampleIds = root.putArray("sample_ids");
            for (Long sampleId : manifest.sampleIds) {
                sampleIds.add(sampleId.longValue());
            }
        }
        root.put("sample_id_col", manifest.sampleIdCol);
        root.put("feature_id_col", manifest.featureIdCol);
        root.put("value_col", manifest.valueCol);
        JsonUtils.writeJson(path, root);
    }

    public static ScalarSampleMajorManifest read(String path) throws IOException {
        JsonNode root = JsonUtils.readJson(path);
        String format = textOrEmpty(root, "format");
        if (!FORMAT.equals(format)) {
            throw new IOException("unsupported sample-major manifest format: " + format);
        }
        ArrayList<String> samplePaths = new ArrayList<String>();
        JsonNode rawSamplePaths = root.get("sample_paths");
        if (rawSamplePaths != null && rawSamplePaths.isArray()) {
            for (JsonNode item : rawSamplePaths) {
                samplePaths.add(resolveAgainst(path, item.asText()));
            }
        }
        ArrayList<Long> sampleIds = null;
        JsonNode rawSampleIds = root.get("sample_ids");
        if (rawSampleIds != null && rawSampleIds.isArray()) {
            sampleIds = new ArrayList<Long>();
            for (JsonNode item : rawSampleIds) {
                sampleIds.add(Long.valueOf(item.asLong()));
            }
        }
        return new ScalarSampleMajorManifest(
                resolveAgainst(path, textOrEmpty(root, "sample_meta_path")),
                resolveAgainst(path, textOrEmpty(root, "feature_meta_path")),
                samplePaths,
                sampleIds,
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
