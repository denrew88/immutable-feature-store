package fs.io;

import fs.model.ShardManifest;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ManifestIO {
    public static void write(ShardManifest manifest, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"sample_meta_path\": \"").append(escapeJson(manifest.sampleMetaPath)).append("\",\n");
        sb.append("  \"n_samples\": ").append(manifest.nSamples).append(",\n");
        sb.append("  \"shard_path\": \"").append(escapeJson(manifest.shardPath)).append("\",\n");
        sb.append("  \"n_shards\": ").append(manifest.nShards).append(",\n");
        sb.append("  \"feature_locator_path\": \"").append(escapeJson(manifest.featureLocatorPath)).append("\",\n");
        sb.append("  \"feature_locator_format\": \"").append(escapeJson(manifest.featureLocatorFormat)).append("\",\n");
        sb.append("  \"feature_id_dtype\": \"").append(escapeJson(manifest.featureIdType)).append("\",\n");
        sb.append("  \"values_dtype\": \"").append(escapeJson(manifest.valuesType)).append("\",\n");
        sb.append("  \"valid_dtype\": \"").append(escapeJson(manifest.validType)).append("\",\n");
        if (manifest.statsYCol != null) {
            sb.append("  \"stats_y_col\": \"").append(escapeJson(manifest.statsYCol)).append("\"\n");
        } else {
            sb.append("  \"stats_y_col\": null\n");
        }
        sb.append("}\n");
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(sb.toString());
        }
    }

    public static ShardManifest read(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        String json = sb.toString();
        String sampleMetaPath = extractString(json, "sample_meta_path");
        int nSamples = extractInt(json, "n_samples");
        String shardPath = extractString(json, "shard_path");
        int nShards = extractInt(json, "n_shards");
        String featureLocatorPath = extractString(json, "feature_locator_path");
        String featureLocatorFormat = extractString(json, "feature_locator_format");
        String featureIdType = extractString(json, "feature_id_dtype");
        String valuesType = extractString(json, "values_dtype");
        String validType = extractString(json, "valid_dtype");
        String statsYCol = extractOptionalString(json, "stats_y_col");
        return new ShardManifest(
                sampleMetaPath,
                nSamples,
                shardPath,
                nShards,
                featureLocatorPath,
                featureLocatorFormat,
                featureIdType,
                valuesType,
                validType,
                statsYCol);
    }

    private static String escapeJson(String s) {
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

    private static int extractInt(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(\\d+)", Pattern.DOTALL);
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("Missing key: " + key);
        }
        return Integer.parseInt(m.group(1));
    }

    private static String extractOptionalString(String json, String key) {
        Pattern nullPattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*null", Pattern.DOTALL);
        Matcher nullMatcher = nullPattern.matcher(json);
        if (nullMatcher.find()) {
            return null;
        }
        Pattern stringPattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"", Pattern.DOTALL);
        Matcher stringMatcher = stringPattern.matcher(json);
        if (stringMatcher.find()) {
            return unescapeJson(stringMatcher.group(1));
        }
        return null;
    }

}
