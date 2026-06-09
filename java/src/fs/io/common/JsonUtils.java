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
     * @throws IOException 파일을 읽거나 JSON을 파싱하지 못한 경우
     */
    public static JsonNode readJson(String path) throws IOException {
        return MAPPER.readTree(new File(path));
    }

    /**
     * JSON tree를 UTF-8 파일로 쓴다.
     *
     * @param path 대상 파일 경로
     * @param node 기록할 JSON tree
     * @throws IOException parent directory 생성 또는 JSON write가 실패한 경우
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
     *
     * @param path 최종 JSON 파일 경로입니다.
     * @param node 저장할 JSON tree입니다.
     * @throws IOException lock 획득, tmp write, final replace, lock release 중 실패한 경우
     */
    public static void writeJsonAtomic(String path, JsonNode node) throws IOException {
        File file = new File(path);
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        File lockFile = new File(path + ".lock");
        File tmp = new File(path + "." + UUID.randomUUID().toString() + ".tmp");
        FilePathLock lock = new FilePathLock(lockFile.getAbsolutePath(), 30000L);
        // 같은 JSON을 두 writer가 동시에 갱신하면 마지막 replace 순서만 남습니다.
        // 따라서 JSON 단위로 lock을 잡고, lock을 잡은 writer만 tmp -> final 교체를 합니다.
        lock.acquire();
        try {
            // tmp 파일명에 UUID를 넣어 이전 실행의 stale tmp나 다른 writer의 tmp와
            // 충돌하지 않게 합니다. reader는 final path만 보므로 partial JSON을 보지 않습니다.
            MAPPER.writeValue(tmp, node);
            moveJsonTmpWithRetry(tmp, file);
        } finally {
            lock.release();
            if (tmp.exists() && !tmp.delete()) {
                tmp.deleteOnExit();
            }
        }
    }

    private static void moveJsonTmpWithRetry(File tmp, File file) throws IOException {
        // 가능한 파일 시스템에서는 atomic move를 먼저 시도합니다. 지원하지 않는 경우에도
        // 같은 lock 안에서 replace하므로 writer 간 충돌은 막을 수 있습니다.
        // Windows에서는 백신/IDE가 final JSON을 잠깐 열어 replace가 실패할 수 있어
        // 짧은 backoff로 재시도합니다.
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
     *
     * @param path JSONL 파일 경로입니다.
     * @param node 한 줄로 append할 JSON object입니다.
     * @throws IOException parent directory 생성, JSON 직렬화, 파일 append가 실패한 경우
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
     *
     * @param path JSONL 파일 경로입니다. 파일이 없으면 빈 목록을 반환합니다.
     * @return 각 유효한 JSONL line을 파싱한 node 목록입니다.
     * @throws IOException 파일을 읽는 중 I/O 오류가 난 경우
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
     *
     * @return 같은 ObjectMapper 설정을 쓰는 빈 object node입니다.
     */
    public static ObjectNode objectNode() {
        return MAPPER.createObjectNode();
    }

    /**
     * 빈 array node를 만든다.
     *
     * @return 같은 ObjectMapper 설정을 쓰는 빈 array node입니다.
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
     * @throws IOException dictionary 파일을 읽지 못했거나 지원하지 않는 구조인 경우
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
     * @throws IOException dictionary JSON write가 실패한 경우
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
