package fs.io;

import fs.model.ScalarSampleBundleManifest;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ScalarSampleBundleManifestIO {
    private ScalarSampleBundleManifestIO() {
    }

    public static void write(ScalarSampleBundleManifest manifest, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"format\": \"scalar-sample-bundles\",\n");
        sb.append("  \"version\": 1,\n");
        sb.append("  \"sample_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.sampleMetaPath))).append("\",\n");
        sb.append("  \"feature_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.featureMetaPath))).append("\",\n");
        sb.append("  \"bundle_paths\": [\n");
        for (int i = 0; i < manifest.bundlePaths.size(); i++) {
            sb.append("    \"").append(escapeJson(relativeTo(path, manifest.bundlePaths.get(i)))).append("\"");
            if (i + 1 < manifest.bundlePaths.size()) {
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append("  ],\n");
        sb.append("  \"sample_id_col\": \"").append(escapeJson(manifest.sampleIdCol)).append("\",\n");
        sb.append("  \"feature_id_col\": \"").append(escapeJson(manifest.featureIdCol)).append("\",\n");
        sb.append("  \"value_col\": \"").append(escapeJson(manifest.valueCol)).append("\"\n");
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    @SuppressWarnings("unchecked")
    public static ScalarSampleBundleManifest read(String path) throws IOException {
        String json = readAll(path);
        try {
            ScriptEngine engine = new ScriptEngineManager().getEngineByName("javascript");
            if (engine == null) {
                throw new IOException("javascript engine is unavailable");
            }
            Object parsed = engine.eval("Java.asJSONCompatible(" + json + ")");
            Map<String, Object> root = (Map<String, Object>) parsed;
            String format = stringValue(root.get("format"));
            if (!"scalar-sample-bundles".equals(format)) {
                throw new IOException("unsupported sample-bundle manifest format: " + format);
            }
            List<Object> rawBundlePaths = (List<Object>) root.get("bundle_paths");
            ArrayList<String> bundlePaths = new ArrayList<String>();
            if (rawBundlePaths != null) {
                for (Object item : rawBundlePaths) {
                    bundlePaths.add(resolveAgainst(path, stringValue(item)));
                }
            }
            return new ScalarSampleBundleManifest(
                    resolveAgainst(path, stringValue(root.get("sample_meta_path"))),
                    resolveAgainst(path, stringValue(root.get("feature_meta_path"))),
                    bundlePaths,
                    defaultString(root.get("sample_id_col"), "sample_id"),
                    defaultString(root.get("feature_id_col"), "feature_id"),
                    defaultString(root.get("value_col"), "value"));
        } catch (Exception e) {
            throw new IOException("failed to parse scalar sample-bundle manifest: " + path, e);
        }
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

    private static String stringValue(Object value) {
        return (value == null) ? "" : value.toString();
    }

    private static String defaultString(Object value, String fallback) {
        String out = stringValue(value);
        return out.isEmpty() ? fallback : out;
    }

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
