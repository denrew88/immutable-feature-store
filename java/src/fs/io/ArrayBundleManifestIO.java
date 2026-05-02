package fs.io;

import fs.model.ArrayBundleManifest;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
        sb.append("  \"time_dtype\": \"").append(escapeJson(manifest.timeType)).append("\",\n");
        sb.append("  \"value_dtype\": \"").append(escapeJson(manifest.valueType)).append("\"\n");
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    public static ArrayBundleManifest read(String path) throws IOException {
        String json = readAll(path);
        return new ArrayBundleManifest(
                resolveAgainst(path, extractString(json, "sample_meta_path")),
                resolveAgainst(path, extractOptionalString(json, "feature_meta_path")),
                extractInt(json, "n_samples"),
                resolveAgainst(path, extractString(json, "bundle_path")),
                extractInt(json, "n_bundles"),
                extractString(json, "feature_id_dtype"),
                extractString(json, "flags_dtype"),
                extractString(json, "time_dtype"),
                extractString(json, "value_dtype"));
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
        File path = new File(storedPath);
        if (path.isAbsolute()) {
            return path.getAbsolutePath();
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

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String unescapeJson(String s) {
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static String extractString(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"", Pattern.DOTALL);
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("Missing key: " + key);
        }
        return unescapeJson(m.group(1));
    }

    private static String extractOptionalString(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"", Pattern.DOTALL);
        Matcher m = p.matcher(json);
        if (!m.find()) {
            return "";
        }
        return unescapeJson(m.group(1));
    }

    private static int extractInt(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(\\d+)", Pattern.DOTALL);
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("Missing key: " + key);
        }
        return Integer.parseInt(m.group(1));
    }
}
