package fs.io.common;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

/**
 * 자바 쪽 JSON read/write를 모아 두는 공통 helper다.
 *
 * <p>manifest와 categorical dictionary는 구조가 작고 고정적이라 tree model만으로 충분하다.
 * 이 helper는 ObjectMapper 생성, 파일 read/write, 기본 node 생성 같은 반복 코드를 한곳에 모은다.
 */
public final class JsonUtils {
    private static final ObjectMapper MAPPER = createMapper();
    private static final int ATOMIC_MOVE_RETRY_COUNT = 8;

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
     * JSON tree를 임시 파일에 쓴 뒤 rename해서 원자적으로 교체한다.
     *
     * <p>Windows에서는 백신, IDE refresh, 이전 디버그 프로세스가 JSON 파일을
     * 짧게 열고 있어도 replace가 실패할 수 있다. 그래서 writer끼리는
     * {@code .lock} 파일 생성으로 직렬화하고, 임시 파일명은 호출마다 다르게 만들어
     * {@code *.tmp} 이름 충돌을 피한다. 최종 move는 짧게 retry한다.</p>
     */
    public static void writeJsonAtomic(String path, JsonNode node) throws IOException {
        File file = new File(path);
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        File lockFile = new File(path + ".lock");
        File tmp = new File(path + "." + UUID.randomUUID().toString() + ".tmp");
        acquireCreateFileLock(lockFile);
        try {
            MAPPER.writeValue(tmp, node);
            moveJsonTmpWithRetry(tmp, file);
        } finally {
            releaseCreateFileLock(lockFile);
            if (tmp.exists() && !tmp.delete()) {
                tmp.deleteOnExit();
            }
        }
    }

    private static void acquireCreateFileLock(File lockFile) throws IOException {
        File parent = lockFile.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create lock directory: " + parent.getAbsolutePath());
        }
        long deadline = System.currentTimeMillis() + 30000L;
        while (true) {
            if (lockFile.createNewFile()) {
                try (FileOutputStream out = new FileOutputStream(lockFile, false)) {
                    out.write(Thread.currentThread().getName().getBytes(StandardCharsets.UTF_8));
                }
                return;
            }
            if (System.currentTimeMillis() >= deadline) {
                throw new IOException("timed out acquiring JSON file lock: " + lockFile.getAbsolutePath());
            }
            try {
                Thread.sleep(50L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted while acquiring JSON file lock: " + lockFile.getAbsolutePath(), e);
            }
        }
    }

    private static void releaseCreateFileLock(File lockFile) {
        if (lockFile.exists() && !lockFile.delete()) {
            // best-effort cleanup; a stale lock will be surfaced by the next acquire timeout.
        }
    }

    private static void moveJsonTmpWithRetry(File tmp, File file) throws IOException {
        IOException last = null;
        for (int attempt = 0; attempt < ATOMIC_MOVE_RETRY_COUNT; attempt++) {
            try {
                try {
                    Files.move(tmp.toPath(), file.toPath(),
                            StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE);
                } catch (AtomicMoveNotSupportedException e) {
                    Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING);
                }
                return;
            } catch (IOException e) {
                last = e;
                if (attempt == ATOMIC_MOVE_RETRY_COUNT - 1) {
                    break;
                }
                try {
                    Thread.sleep(25L * (attempt + 1));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw e;
                }
            }
        }
        throw last;
    }

    /**
     * JSON object 한 줄을 JSONL 파일 끝에 append한다.
     */
    public static void appendJsonLine(String path, JsonNode node) throws IOException {
        File file = new File(path);
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        String line = MAPPER.writer().without(SerializationFeature.INDENT_OUTPUT).writeValueAsString(node);
        try (OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(file, true), StandardCharsets.UTF_8)) {
            writer.write(line);
            writer.write(System.lineSeparator());
        }
    }

    /**
     * JSONL 파일을 한 줄씩 읽어 JsonNode 목록으로 돌려준다.
     */
    public static List<JsonNode> readJsonLines(String path) throws IOException {
        ArrayList<JsonNode> out = new ArrayList<JsonNode>();
        File file = new File(path);
        if (!file.exists()) {
            return out;
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String rawLine;
            while ((rawLine = reader.readLine()) != null) {
                String line = rawLine.trim();
                if (line.isEmpty()) {
                    continue;
                }
                try {
                    out.add(MAPPER.readTree(line));
                } catch (IOException ignoredTail) {
                    break;
                }
            }
        }
        return out;
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
