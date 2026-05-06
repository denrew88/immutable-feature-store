package fs.io.common;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

/**
 * 자바 쪽 JSON read/write를 모아 두는 공통 helper다.
 *
 * <p>manifest와 categorical dictionary는 구조가 작고 고정적이라 tree model만으로 충분하다.
 * 이 helper는 ObjectMapper 생성, 파일 read/write, 기본 node 생성 같은 반복 코드를 한곳에 모은다.
 */
public final class JsonUtils {
    private static final ObjectMapper MAPPER = createMapper();

    private JsonUtils() {
    }

    /**
     * JSON 파일을 tree model로 읽는다.
     *
     * @param path JSON 파일 경로
     * @return 파싱된 root node
     */
    public static JsonNode readJson(String path) throws IOException {
        return MAPPER.readTree(new File(path));
    }

    /**
     * JSON tree를 UTF-8 파일로 쓴다.
     *
     * @param path 대상 파일 경로
     * @param node 기록할 JSON tree
     */
    public static void writeJson(String path, JsonNode node) throws IOException {
        File file = new File(path);
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        MAPPER.writeValue(file, node);
    }

    /**
     * 빈 object node를 만든다.
     */
    public static ObjectNode objectNode() {
        return MAPPER.createObjectNode();
    }

    /**
     * 빈 array node를 만든다.
     */
    public static ArrayNode arrayNode() {
        return MAPPER.createArrayNode();
    }

    /**
     * categorical dictionary JSON을 code와 label의 대응 map으로 읽는다.
     *
     * <p>현재 포맷은 {"column": ..., "items": [{"code": ..., "label": ...}, ...]}를 기본으로 쓰지만,
     * 예전 {"labels": {"1": "A", ...}} 형태도 함께 받아들인다.
     *
     * @param path dictionary JSON 경로
     * @return code와 label의 대응 map
     */
    public static HashMap<Long, String> readCategoricalDictionary(String path) throws IOException {
        JsonNode root = readJson(path);
        if (!root.isObject()) {
            throw new IOException("categorical dictionary root must be an object: " + path);
        }

        HashMap<Long, String> out = new HashMap<Long, String>();
        JsonNode labels = root.get("labels");
        if (labels != null && labels.isObject()) {
            java.util.Iterator<String> fieldNames = labels.fieldNames();
            while (fieldNames.hasNext()) {
                String codeText = fieldNames.next();
                JsonNode labelNode = labels.get(codeText);
                out.put(Long.valueOf(codeText), labelNode == null || labelNode.isNull() ? null : labelNode.asText());
            }
            return out;
        }

        JsonNode items = root.get("items");
        if (items != null && items.isArray()) {
            for (JsonNode item : items) {
                if (!item.isObject() || !item.has("code") || !item.has("label")) {
                    throw new IOException("categorical dictionary items must contain code/label: " + path);
                }
                JsonNode labelNode = item.get("label");
                out.put(item.get("code").asLong(), labelNode == null || labelNode.isNull() ? null : labelNode.asText());
            }
            return out;
        }

        throw new IOException("unsupported categorical dictionary JSON structure: " + path);
    }

    /**
     * categorical dictionary JSON을 현재 표준 포맷으로 쓴다.
     *
     * @param path 대상 JSON 경로
     * @param columnName categorical point column 이름
     * @param labels code 1..N에 대응하는 label 목록
     */
    public static void writeCategoricalDictionary(String path, String columnName, List<String> labels) throws IOException {
        ObjectNode root = objectNode();
        root.put("column", columnName);
        ArrayNode items = root.putArray("items");
        for (int i = 0; i < labels.size(); i++) {
            ObjectNode item = items.addObject();
            item.put("code", i + 1);
            String label = labels.get(i);
            if (label == null) {
                item.putNull("label");
            } else {
                item.put("label", label);
            }
        }
        writeJson(path, root);
    }

    private static ObjectMapper createMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        return mapper;
    }
}
