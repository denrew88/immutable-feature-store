package fs.io.scalar;

import fs.model.scalar.ShardManifest;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Scalar shard manifest JSON을 읽고 쓰는 helper다.
 */
public class ManifestIO {
    public static void write(ShardManifest manifest, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"sample_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.sampleMetaPath))).append("\",\n");
        sb.append("  \"feature_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.featureMetaPath))).append("\",\n");
        sb.append("  \"n_samples\": ").append(manifest.nSamples).append(",\n");
        sb.append("  \"n_features\": ").append(manifest.nFeatures).append(",\n");
        sb.append("  \"shard_path\": \"").append(escapeJson(relativeTo(path, manifest.shardPath))).append("\",\n");
        sb.append("  \"n_shards\": ").append(manifest.nShards).append(",\n");
        sb.append("  \"feature_locator_path\": \"").append(escapeJson(relativeTo(path, manifest.featureLocatorPath))).append("\",\n");
        sb.append("  \"feature_locator_format\": \"").append(escapeJson(manifest.featureLocatorFormat)).append("\",\n");
        sb.append("  \"feature_id_dtype\": \"").append(escapeJson(manifest.featureIdType)).append("\",\n");
        sb.append("  \"values_dtype\": \"").append(escapeJson(manifest.valuesType)).append("\",\n");
        sb.append("  \"valid_dtype\": \"").append(escapeJson(manifest.validType)).append("\",\n");
        sb.append("  \"id_scheme\": \"").append(escapeJson(manifest.idScheme)).append("\",\n");
        sb.append("  \"sample_key_col\": \"").append(escapeJson(manifest.sampleKeyCol)).append("\",\n");
        sb.append("  \"feature_key_col\": \"").append(escapeJson(manifest.featureKeyCol)).append("\"");
        if (manifest.targetShardBytes != null) {
            sb.append(",\n");
            sb.append("  \"target_shard_bytes\": ").append(manifest.targetShardBytes.longValue());
        }
        if (!manifest.selectionStats.isEmpty()) {
            sb.append(",\n");
            sb.append("  \"selection_stats\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : manifest.selectionStats.entrySet()) {
                sb.append("    \"").append(escapeJson(entry.getKey())).append("\": ");
                sb.append("\"").append(escapeJson(relativeTo(path, entry.getValue()))).append("\"");
                if (i + 1 < manifest.selectionStats.size()) {
                    sb.append(",");
                }
                sb.append("\n");
                i++;
            }
            sb.append("  }");
        } else if (manifest.statsYCol != null) {
            sb.append(",\n");
            sb.append("  \"stats_y_col\": \"").append(escapeJson(manifest.statsYCol)).append("\"");
        }
        sb.append("\n");
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    @SuppressWarnings("unchecked")
    public static ShardManifest read(String path) throws IOException {
        String json = readAll(path);
        try {
            ScriptEngine engine = new ScriptEngineManager().getEngineByName("javascript");
            if (engine == null) {
                throw new IOException("javascript engine is unavailable");
            }
            Object parsed = engine.eval("Java.asJSONCompatible(" + json + ")");
            Map<String, Object> root = (Map<String, Object>) parsed;
            return new ShardManifest(
                    resolveAgainst(path, stringValue(root.get("sample_meta_path"))),
                    resolveAgainst(path, stringValue(root.get("feature_meta_path"))),
                    intValue(root.get("n_samples")),
                    intValue(root.get("n_features")),
                    resolveAgainst(path, stringValue(root.get("shard_path"))),
                    intValue(root.get("n_shards")),
                    resolveAgainst(path, stringValue(root.get("feature_locator_path"))),
                    stringValue(root.get("feature_locator_format")),
                    stringValue(root.get("feature_id_dtype")),
                    stringValue(root.get("values_dtype")),
                    stringValue(root.get("valid_dtype")),
                    defaultString(root.get("id_scheme"), "legacy"),
                    defaultString(root.get("sample_key_col"), "sample_key"),
                    defaultString(root.get("feature_key_col"), "feature_key"),
                    longObject(root.get("target_shard_bytes")),
                    parseSelectionStats(path, (Map<String, Object>) root.get("selection_stats")),
                    optionalString(root.get("stats_y_col"))
            );
        } catch (Exception e) {
            throw new IOException("failed to parse shard manifest: " + path, e);
        }
    }

    private static Map<String, String> parseSelectionStats(String manifestPath, Map<String, Object> raw) {
        if (raw == null || raw.isEmpty()) {
            return Collections.emptyMap();
        }
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        for (Map.Entry<String, Object> entry : raw.entrySet()) {
            out.put(entry.getKey(), resolveAgainst(manifestPath, stringValue(entry.getValue())));
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

    private static int intValue(Object value) {
        return (value == null) ? 0 : ((Number) value).intValue();
    }

    private static Long longObject(Object value) {
        return (value == null) ? null : Long.valueOf(((Number) value).longValue());
    }

    private static String stringValue(Object value) {
        return (value == null) ? "" : value.toString();
    }

    private static String optionalString(Object value) {
        return (value == null) ? null : value.toString();
    }

    private static String defaultString(Object value, String fallback) {
        String out = optionalString(value);
        return (out == null || out.isEmpty()) ? fallback : out;
    }

    private static String readAll(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
