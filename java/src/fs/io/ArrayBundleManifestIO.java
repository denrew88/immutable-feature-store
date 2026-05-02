package fs.io;

import fs.model.ArrayBundleManifest;
import fs.model.LogicalType;
import fs.model.PointColumnSpec;
import fs.model.StorageType;

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

public class ArrayBundleManifestIO {
    public static void write(ArrayBundleManifest manifest, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"sample_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.sampleMetaPath))).append("\",\n");
        sb.append("  \"feature_meta_path\": \"").append(escapeJson(relativeTo(path, manifest.featureMetaPath))).append("\",\n");
        sb.append("  \"n_samples\": ").append(manifest.nSamples).append(",\n");
        sb.append("  \"bundle_path\": \"").append(escapeJson(relativeTo(path, manifest.bundlePath))).append("\",\n");
        sb.append("  \"n_bundles\": ").append(manifest.nBundles).append(",\n");
        sb.append("  \"feature_id_dtype\": \"").append(escapeJson(manifest.featureIdType)).append("\",\n");
        sb.append("  \"flags_dtype\": \"").append(escapeJson(manifest.flagsType)).append("\",\n");
        if (manifest.timeType != null && !manifest.timeType.isEmpty()) {
            sb.append("  \"time_dtype\": \"").append(escapeJson(manifest.timeType)).append("\",\n");
        }
        if (manifest.valueType != null && !manifest.valueType.isEmpty()) {
            sb.append("  \"value_dtype\": \"").append(escapeJson(manifest.valueType)).append("\",\n");
        }
        sb.append("  \"point_schema\": [\n");
        for (int i = 0; i < manifest.pointSchema.size(); i++) {
            PointColumnSpec spec = manifest.pointSchema.get(i);
            sb.append("    {\n");
            sb.append("      \"name\": \"").append(escapeJson(spec.name)).append("\",\n");
            sb.append("      \"storage_type\": \"").append(escapeJson(spec.storageType.value)).append("\",\n");
            sb.append("      \"logical_type\": \"").append(escapeJson(spec.logicalType.value)).append("\"");
            if (spec.dictionaryPath != null && !spec.dictionaryPath.isEmpty()) {
                sb.append(",\n");
                sb.append("      \"dictionary_path\": \"").append(escapeJson(relativeTo(path, spec.dictionaryPath))).append("\"\n");
            } else {
                sb.append("\n");
            }
            sb.append("    }");
            if (i + 1 < manifest.pointSchema.size()) {
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    @SuppressWarnings("unchecked")
    public static ArrayBundleManifest read(String path) throws IOException {
        String json = readAll(path);
        try {
            ScriptEngine engine = new ScriptEngineManager().getEngineByName("javascript");
            if (engine == null) {
                throw new IOException("javascript engine is unavailable");
            }
            Object parsed = engine.eval("Java.asJSONCompatible(" + json + ")");
            Map<String, Object> root = (Map<String, Object>) parsed;
            return new ArrayBundleManifest(
                    resolveAgainst(path, stringValue(root.get("sample_meta_path"))),
                    resolveAgainst(path, stringValue(root.get("feature_meta_path"))),
                    intValue(root.get("n_samples")),
                    resolveAgainst(path, stringValue(root.get("bundle_path"))),
                    intValue(root.get("n_bundles")),
                    stringValue(root.get("feature_id_dtype")),
                    stringValue(root.get("flags_dtype")),
                    stringValue(root.get("time_dtype")),
                    stringValue(root.get("value_dtype")),
                    parsePointSchema(path, (List<Object>) root.get("point_schema")));
        } catch (Exception e) {
            throw new IOException("failed to parse array bundle manifest: " + path, e);
        }
    }

    private static List<PointColumnSpec> parsePointSchema(String manifestPath, List<Object> raw) {
        if (raw == null || raw.isEmpty()) {
            return PointColumnSpec.defaultPointSchema();
        }
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(raw.size());
        for (Object item : raw) {
            Map<String, Object> data = (Map<String, Object>) item;
            out.add(new PointColumnSpec(
                    stringValue(data.get("name")),
                    StorageType.fromValue(stringValue(data.get("storage_type"))),
                    LogicalType.fromValue(stringValue(data.get("logical_type"))),
                    resolveAgainst(manifestPath, stringValue(data.get("dictionary_path")))));
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

    private static int intValue(Object value) {
        return (value == null) ? 0 : ((Number) value).intValue();
    }

    private static String stringValue(Object value) {
        return (value == null) ? "" : value.toString();
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
