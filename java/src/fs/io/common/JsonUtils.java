package fs.io.common;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Array categorical dictionary JSON을 읽고 쓰는 공통 helper다.
 */
public final class JsonUtils {
    private JsonUtils() {
    }

    static Object readJsonFile(String path) throws IOException {
        return parseJson(new String(Files.readAllBytes(new File(path).toPath()), StandardCharsets.UTF_8), path);
    }

    public static HashMap<Long, String> readCategoricalDictionary(String path) throws IOException {
        Object parsed = readJsonFile(path);
        if (!(parsed instanceof Map<?, ?>)) {
            throw new IOException("categorical dictionary root must be an object: " + path);
        }
        Map<?, ?> root = (Map<?, ?>) parsed;
        HashMap<Long, String> out = new HashMap<Long, String>();
        Object labels = root.get("labels");
        if (labels instanceof Map<?, ?>) {
            Map<?, ?> mapping = (Map<?, ?>) labels;
            for (Map.Entry<?, ?> entry : mapping.entrySet()) {
                out.put(Long.parseLong(String.valueOf(entry.getKey())), entry.getValue() == null ? null : String.valueOf(entry.getValue()));
            }
            return out;
        }
        Object items = root.get("items");
        if (items instanceof List<?>) {
            for (Object item : (List<?>) items) {
                if (!(item instanceof Map<?, ?>)) {
                    throw new IOException("categorical dictionary items must be objects: " + path);
                }
                Map<?, ?> row = (Map<?, ?>) item;
                Object code = row.get("code");
                if (code == null || !row.containsKey("label")) {
                    throw new IOException("categorical dictionary items must contain code/label: " + path);
                }
                Object label = row.get("label");
                out.put(Long.valueOf(String.valueOf(code)), label == null ? null : String.valueOf(label));
            }
            return out;
        }
        throw new IOException("unsupported categorical dictionary JSON structure: " + path);
    }

    public static void writeCategoricalDictionary(String path, String columnName, List<String> labels) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"column\": \"").append(escapeJson(columnName)).append("\",\n");
        sb.append("  \"items\": [\n");
        for (int i = 0; i < labels.size(); i++) {
            sb.append("    {\"code\": ").append(i + 1).append(", \"label\": ");
            String label = labels.get(i);
            if (label == null) {
                sb.append("null");
            } else {
                sb.append("\"").append(escapeJson(label)).append("\"");
            }
            sb.append("}");
            if (i + 1 < labels.size()) {
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        Files.write(new File(path).toPath(), sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static Object parseJson(String json, String label) throws IOException {
        ScriptEngine engine = new ScriptEngineManager().getEngineByName("javascript");
        if (engine == null) {
            throw new IOException("javascript engine not available while parsing JSON: " + label);
        }
        try {
            engine.put("__json_text__", json);
            return engine.eval("Java.asJSONCompatible(JSON.parse(__json_text__))");
        } catch (Exception e) {
            throw new IOException("failed to parse JSON: " + label, e);
        }
    }

    private static String escapeJson(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            switch (ch) {
                case '\\':
                    sb.append("\\\\");
                    break;
                case '"':
                    sb.append("\\\"");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    sb.append(ch);
                    break;
            }
        }
        return sb.toString();
    }
}
