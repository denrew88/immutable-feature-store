package fs.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.config.BuildShardConfig;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.JsonUtils;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.io.scalar.ScalarFileLock;
import fs.io.scalar.ScalarMetadataWriter;
import fs.io.scalar.ScalarRawSampleWriter;
import fs.io.scalar.ScalarSampleMajorManifestIO;
import fs.model.scalar.ScalarBuildSessionStatus;
import fs.model.scalar.ScalarSampleMajorManifest;

import java.io.File;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * scalar dense-long shard를 만들기 위한 표준 builder입니다.
 *
 * <p>sample 하나가 정상 종료될 때마다 {@code raw_samples/sample_*.parquet}
 * 파일 하나를 commit합니다. sample id 순서를 강제하지 않으므로 여러 worker가
 * 서로 다른 sample id를 나눠 처리할 수 있습니다. 순차 실행을 원하면
 * {@link #status()}의 {@code pendingSampleIds}를 앞에서부터 처리하면 됩니다.</p>
 */
public class ScalarDatasetBuilder implements AutoCloseable {
    private static final int RAW_STATE_VERSION = 1;
    private static final int RAW_SAMPLE_PADDING = 12;
    private static final int FILE_OPERATION_RETRY_COUNT = 10;
    private static final long FILE_OPERATION_RETRY_BASE_MILLIS = 25L;

    private final File outDir;
    private final File rawSamplesDir;
    private final String sampleMetaSourcePath;
    private final String featureMetaSourcePath;
    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final String rawLogPath;
    private final String statePath;
    private final String sampleMajorManifestPath;
    private final BuildShardConfig buildConfig;
    private final int nSamples;
    private final ArrayList<String> sampleKeys;
    private final LinkedHashMap<String, Integer> featureKeyToId = new LinkedHashMap<String, Integer>();
    private final ArrayList<String> featureKeysInOrder = new ArrayList<String>();

    private boolean finishedStage;
    private boolean closed;

    public ScalarDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        if ((featureMetaPath == null || featureMetaPath.isEmpty()) && featureKeys == null) {
            throw new IllegalArgumentException("ScalarDatasetBuilder requires featureMetaPath or featureKeys");
        }
        this.outDir = new File(outDir).getAbsoluteFile();
        this.rawSamplesDir = new File(this.outDir, "raw_samples");
        this.sampleMetaSourcePath = new File(sampleMetaPath).getAbsolutePath();
        this.featureMetaSourcePath = featureMetaPath == null || featureMetaPath.isEmpty()
                ? ""
                : new File(featureMetaPath).getAbsolutePath();
        this.sampleMetaPath = new File(this.outDir, "sample_meta.parquet").getAbsolutePath();
        this.featureMetaPath = new File(this.outDir, "feature_meta.parquet").getAbsolutePath();
        this.rawLogPath = new File(this.outDir, "raw_samples.jsonl").getAbsolutePath();
        this.statePath = new File(this.outDir, "raw_state.json").getAbsolutePath();
        this.sampleMajorManifestPath = new File(this.outDir, "sample_major_manifest.json").getAbsolutePath();
        this.buildConfig = buildConfig == null ? new BuildShardConfig() : buildConfig;

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sampleMetaSourcePath);
        this.nSamples = sampleRows.size();
        this.sampleKeys = loadKeys(sampleRows, this.buildConfig.sampleKeyCol);

        if (new File(this.statePath).exists()) {
            resume(featureMetaPath, featureKeys);
        } else {
            initialize(featureMetaPath, featureKeys);
        }
    }

    public static ScalarDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig);
    }

    public ScalarBuildSessionStatus status() throws Exception {
        List<Long> completed = completedSampleIds();
        BitSet done = new BitSet(nSamples);
        for (Long sampleId : completed) {
            long value = sampleId.longValue();
            if (value >= 0L && value < nSamples) {
                done.set((int) value);
            }
        }
        ArrayList<Long> pending = new ArrayList<Long>();
        for (int sampleId = 0; sampleId < nSamples; sampleId++) {
            if (!done.get(sampleId)) {
                pending.add(Long.valueOf(sampleId));
            }
        }
        return new ScalarBuildSessionStatus(nSamples, completed, pending, finishedStage, finishedStage ? sampleMajorManifestPath : "");
    }

    public List<Long> pendingSampleIds() throws Exception {
        return status().pendingSampleIds;
    }

    /**
     * sample 하나를 raw parquet 파일로 commit합니다.
     *
     * <p>{@code values}는 feature id 또는 feature key를 key로 갖는 map입니다.
     * 값이 {@code null}이거나 {@code NaN}이면 missing으로 보고 raw row를 쓰지
     * 않습니다.</p>
     */
    public void writeSample(long sampleId, Map<?, ?> values) throws Exception {
        writeSample(sampleId, values, false);
    }

    /**
     * sample 하나를 raw parquet 파일로 commit합니다.
     *
     * <p>{@code skipIfCompleted}가 true이면 이미 commit된 sample을 조용히
     * 건너뜁니다. worker 재시도 코드에서는 이 값을 true로 두는 것이 보통 안전합니다.</p>
     */
    public void writeSample(long sampleId, Map<?, ?> values, boolean skipIfCompleted) throws Exception {
        ensureOpenForWrites();
        validateSampleId(sampleId);
        if (isSampleCompleted(sampleId)) {
            if (skipIfCompleted) {
                return;
            }
            throw new IllegalArgumentException("sample already completed: " + sampleId);
        }

        ScalarFileLock lock = new ScalarFileLock(rawSamplePath(sampleId) + ".lock");
        lock.acquire();
        try {
            if (isSampleCompleted(sampleId)) {
                if (skipIfCompleted) {
                    return;
                }
                throw new IllegalArgumentException("sample already completed: " + sampleId);
            }
            LinkedHashMap<Integer, Double> normalized = normalizeValues(values);
            String finalPath = rawSamplePath(sampleId);
            cleanupSampleTmpFiles(sampleId);
            File tmpFile = uniqueRawSampleTmpFile(finalPath);
            try {
                int rowCount = ScalarRawSampleWriter.write(tmpFile.getAbsolutePath(), sampleId, normalized);
                moveTmpToFinalWithRetry(tmpFile, new File(finalPath));
                appendCommit(sampleId, finalPath, rowCount);
            } finally {
                deleteQuietly(tmpFile);
            }
        } finally {
            lock.release();
        }
    }

    public String finishStage() throws Exception {
        return finishStage(true);
    }

    /**
     * raw sample 파일 목록을 sample-major manifest로 확정합니다.
     *
     * <p>이 단계는 raw parquet를 다시 쓰지 않고, commit log에 있는 sample 파일
     * 경로를 {@code sample_major_manifest.json}에 연결합니다. dense-long build는
     * 이 manifest를 입력으로 사용합니다.</p>
     */
    public String finishStage(boolean requireAll) throws Exception {
        ensureOpen();
        if (finishedStage) {
            return sampleMajorManifestPath;
        }
        ScalarBuildSessionStatus status = status();
        if (requireAll && !status.pendingSampleIds.isEmpty()) {
            throw new IllegalStateException("cannot finish scalar stage: pending sample count=" + status.pendingSampleIds.size());
        }
        List<Long> completed = completedSampleIds();
        Collections.sort(completed);
        ArrayList<String> paths = new ArrayList<String>(completed.size());
        for (Long sampleId : completed) {
            paths.add(rawSamplePath(sampleId.longValue()));
        }
        ScalarSampleMajorManifestIO.write(
                new ScalarSampleMajorManifest(
                        sampleMetaPath,
                        featureMetaPath,
                        paths,
                        completed,
                        buildConfig.sampleIdCol,
                        buildConfig.featureIdCol,
                        buildConfig.valueCol),
                sampleMajorManifestPath);
        finishedStage = true;
        saveState();
        return sampleMajorManifestPath;
    }

    public String buildDenseLongShards(boolean requireAll, String denseLongOutDir) throws Exception {
        return buildDenseLongShards(requireAll, denseLongOutDir, false);
    }

    /**
     * 현재 raw sample stage에서 최종 dense-long shard를 생성합니다.
     *
     * <p>{@code cleanupRaw}가 true이면 build가 성공한 뒤 큰 용량을 차지하는
     * {@code raw_samples/} parquet 파일들을 삭제합니다. stage manifest, commit log,
     * state, metadata는 작고 감사/재현에 쓸 수 있으므로 남깁니다.</p>
     */
    public String buildDenseLongShards(boolean requireAll, String denseLongOutDir, boolean cleanupRaw) throws Exception {
        String stageManifest = finishStage(requireAll);
        String target = denseLongOutDir == null || denseLongOutDir.isEmpty()
                ? new File(outDir, "scalar_shard").getAbsolutePath()
                : denseLongOutDir;
        String manifestPath = ScalarDenseLongShardBuilder.buildFromSampleMajorManifest(stageManifest, target, buildConfig);
        if (cleanupRaw) {
            deleteRecursively(rawSamplesDir);
        }
        return manifestPath;
    }

    public String buildShards(boolean requireAll) throws Exception {
        return buildDenseLongShards(requireAll, null);
    }

    /**
     * 현재 raw sample stage에서 최종 dense-long shard를 생성하고, 필요하면 raw sample parquet를 삭제합니다.
     *
     * <p>두 번째 인자는 cleanup 여부입니다. 기존 {@code buildShards(false)}의 false는
     * {@code requireAll=false}이므로 의미가 다릅니다.</p>
     */
    public String buildShards(boolean requireAll, boolean cleanupRaw) throws Exception {
        return buildDenseLongShards(requireAll, null, cleanupRaw);
    }

    /**
     * 현재 raw sample stage에서 최종 dense-long shard를 생성합니다.
     *
     * <p>{@code requireAll}이 true이면 모든 sample이 commit되어 있어야 합니다.
     * false이면 완료된 sample만 포함한 shard를 만들 수 있지만, 일반적인 전체 dataset
     * build에서는 true를 권장합니다.</p>
     */
    public String buildShards(boolean requireAll, String denseLongOutDir) throws Exception {
        return buildDenseLongShards(requireAll, denseLongOutDir);
    }

    /**
     * 현재 raw sample stage에서 최종 dense-long shard를 지정 경로에 생성하고,
     * 필요하면 raw sample parquet를 삭제합니다.
     */
    public String buildShards(boolean requireAll, String denseLongOutDir, boolean cleanupRaw) throws Exception {
        return buildDenseLongShards(requireAll, denseLongOutDir, cleanupRaw);
    }

    @Override
    public void close() throws Exception {
        if (!closed) {
            saveState();
            closed = true;
        }
    }

    private void initialize(String featureMetaPath, List<String> featureKeys) throws Exception {
        if (outDir.exists()) {
            String[] children = outDir.list();
            if (children != null && children.length > 0) {
                throw new IllegalArgumentException("outDir already exists and is not empty: " + outDir.getAbsolutePath());
            }
        }
        ensureDir(outDir);
        ensureDir(rawSamplesDir);
        Files.copy(new File(sampleMetaSourcePath).toPath(), new File(sampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
        initializeFeatures(featureMetaPath, featureKeys);
        finishedStage = false;
        saveState();
    }

    private void initializeFeatures(String featureMetaPath, List<String> featureKeys) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            Files.copy(new File(featureMetaSourcePath).toPath(), new File(this.featureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            loadFeatureKeys(ArrayMetadataWriter.readRows(this.featureMetaPath));
            return;
        }
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        for (String key : featureKeys) {
            if (featureKeyToId.containsKey(key)) {
                throw new IllegalArgumentException("duplicate feature key: " + key);
            }
            featureKeyToId.put(key, Integer.valueOf(featureKeysInOrder.size()));
            featureKeysInOrder.add(key);
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            row.put(buildConfig.featureKeyCol, key);
            rows.add(row);
        }
        ScalarMetadataWriter.writeFeatureMeta(rows, this.featureMetaPath);
    }

    private void resume(String featureMetaPath, List<String> featureKeys) throws Exception {
        cleanupTmpFiles();
        JsonNode state = JsonUtils.readJson(statePath);
        if (!"scalar_raw_stage_v1".equals(textOrEmpty(state, "format"))) {
            throw new IOException("unsupported scalar stage format: " + textOrEmpty(state, "format"));
        }
        if (!new File(textOrEmpty(state, "sample_meta_source_path")).getAbsolutePath().equals(sampleMetaSourcePath)) {
            throw new IOException("sampleMetaPath does not match existing scalar stage");
        }
        String expectedFeature = featureMetaPath == null || featureMetaPath.isEmpty() ? "" : new File(featureMetaPath).getAbsolutePath();
        if (!textOrEmpty(state, "feature_meta_source_path").equals(expectedFeature)) {
            throw new IOException("featureMetaPath does not match existing scalar stage");
        }
        finishedStage = state.path("finished_stage").asBoolean(false);
        JsonNode featureKeysNode = state.get("feature_keys_in_order");
        if (featureKeysNode != null && featureKeysNode.isArray()) {
            for (JsonNode item : featureKeysNode) {
                String key = item.asText();
                featureKeyToId.put(key, Integer.valueOf(featureKeysInOrder.size()));
                featureKeysInOrder.add(key);
            }
        } else {
            loadFeatureKeys(ArrayMetadataWriter.readRows(this.featureMetaPath));
        }
        if (featureKeys != null && featureKeys.size() != featureKeysInOrder.size()) {
            throw new IOException("featureKeys do not match existing scalar stage");
        }
    }

    private LinkedHashMap<Integer, Double> normalizeValues(Map<?, ?> values) {
        LinkedHashMap<Integer, Double> out = new LinkedHashMap<Integer, Double>();
        if (values == null) {
            return out;
        }
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            int featureId = resolveFeatureId(entry.getKey());
            Object rawValue = entry.getValue();
            if (rawValue == null) {
                continue;
            }
            double value = ((Number) rawValue).doubleValue();
            if (Double.isNaN(value)) {
                continue;
            }
            out.put(Integer.valueOf(featureId), Double.valueOf(value));
        }
        return out;
    }

    private int resolveFeatureId(Object featureRef) {
        if (featureRef instanceof Number) {
            int featureId = ((Number) featureRef).intValue();
            if (featureId < 0 || featureId >= featureKeysInOrder.size()) {
                throw new IllegalArgumentException("feature_id out of range: " + featureId);
            }
            return featureId;
        }
        String key = String.valueOf(featureRef);
        Integer id = featureKeyToId.get(key);
        if (id == null) {
            throw new IllegalArgumentException("unknown feature key: " + key);
        }
        return id.intValue();
    }

    private void appendCommit(long sampleId, String path, int rowCount) throws IOException {
        ObjectNode node = JsonUtils.objectNode();
        node.put("sample_id", sampleId);
        node.put("sample_key", sampleKeyForId(sampleId));
        node.put("path", relativeTo(outDir.getAbsolutePath(), path));
        node.put("row_count", rowCount);
        ScalarFileLock lock = new ScalarFileLock(rawLogPath + ".lock");
        lock.acquire();
        try {
            JsonUtils.appendJsonLine(rawLogPath, node);
        } finally {
            lock.release();
        }
    }

    private List<Long> completedSampleIds() throws IOException {
        ArrayList<Long> out = new ArrayList<Long>();
        BitSet seen = new BitSet(nSamples);
        for (JsonNode item : JsonUtils.readJsonLines(rawLogPath)) {
            long sampleId = item.path("sample_id").asLong(-1L);
            if (sampleId < 0L || sampleId >= nSamples || seen.get((int) sampleId)) {
                continue;
            }
            String path = resolveAgainst(outDir.getAbsolutePath(), textOrEmpty(item, "path"));
            if (new File(path).exists()) {
                seen.set((int) sampleId);
                out.add(Long.valueOf(sampleId));
            }
        }
        Collections.sort(out);
        return out;
    }

    private boolean isSampleCompleted(long sampleId) throws IOException {
        return completedSampleIds().contains(Long.valueOf(sampleId));
    }

    private void saveState() throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", "scalar_raw_stage_v1");
        root.put("raw_state_version", RAW_STATE_VERSION);
        root.put("sample_meta_source_path", sampleMetaSourcePath);
        root.put("feature_meta_source_path", featureMetaSourcePath);
        root.put("sample_meta_path", sampleMetaPath);
        root.put("feature_meta_path", featureMetaPath);
        root.put("finished_stage", finishedStage);
        root.put("n_samples", nSamples);
        com.fasterxml.jackson.databind.node.ArrayNode keys = root.putArray("feature_keys_in_order");
        for (String key : featureKeysInOrder) {
            keys.add(key);
        }
        JsonUtils.writeJsonAtomic(statePath, root);
    }

    private void loadFeatureKeys(List<LinkedHashMap<String, Object>> featureRows) {
        for (int i = 0; i < featureRows.size(); i++) {
            Object rawKey = featureRows.get(i).get(buildConfig.featureKeyCol);
            String key = rawKey == null ? String.valueOf(i) : rawKey.toString();
            featureKeyToId.put(key, Integer.valueOf(i));
            featureKeysInOrder.add(key);
        }
    }

    private String rawSamplePath(long sampleId) {
        return new File(rawSamplesDir, String.format("sample_%0" + RAW_SAMPLE_PADDING + "d.parquet", Long.valueOf(sampleId))).getAbsolutePath();
    }

    private String sampleKeyForId(long sampleId) {
        if (sampleId < 0L || sampleId >= sampleKeys.size()) {
            return "";
        }
        String key = sampleKeys.get((int) sampleId);
        return key == null ? "" : key;
    }

    private void validateSampleId(long sampleId) {
        if (sampleId < 0L || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("scalar builder is closed");
        }
    }

    private void ensureOpenForWrites() {
        ensureOpen();
        if (finishedStage) {
            throw new IllegalStateException("scalar stage has already been finalized");
        }
    }

    private void cleanupTmpFiles() {
        File[] files = rawSamplesDir.listFiles();
        if (files == null) {
            return;
        }
        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(".tmp")) {
                deleteQuietly(file);
            }
        }
    }

    private void cleanupSampleTmpFiles(long sampleId) {
        String fixedPrefix = new File(rawSamplePath(sampleId)).getName() + ".";
        String legacyTmpName = new File(rawSamplePath(sampleId)).getName() + ".tmp";
        File[] files = rawSamplesDir.listFiles();
        if (files == null) {
            return;
        }
        for (File file : files) {
            String name = file.getName();
            if (file.isFile() && name.endsWith(".tmp") && (name.equals(legacyTmpName) || name.startsWith(fixedPrefix))) {
                deleteQuietly(file);
            }
        }
    }

    private static File uniqueRawSampleTmpFile(String finalPath) {
        return new File(finalPath + "." + UUID.randomUUID().toString() + ".tmp").getAbsoluteFile();
    }

    private static ArrayList<String> loadKeys(List<LinkedHashMap<String, Object>> rows, String keyCol) {
        ArrayList<String> out = new ArrayList<String>(rows.size());
        for (LinkedHashMap<String, Object> row : rows) {
            Object value = row.get(keyCol);
            out.add(value == null ? null : value.toString());
        }
        return out;
    }

    private static void ensureDir(File dir) {
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("failed to create dir: " + dir.getAbsolutePath());
        }
    }

    private static void deleteQuietly(File path) {
        if (path == null || !path.exists()) {
            return;
        }
        try {
            deleteWithRetry(path);
        } catch (IOException ignored) {
            path.deleteOnExit();
        }
    }

    private static void deleteWithRetry(File path) throws IOException {
        IOException last = null;
        for (int attempt = 0; attempt < FILE_OPERATION_RETRY_COUNT; attempt++) {
            if (!path.exists() || path.delete()) {
                return;
            }
            last = new IOException("failed to delete file: " + path.getAbsolutePath());
            sleepBeforeRetry(attempt, last);
        }
        throw last == null ? new IOException("failed to delete file: " + path.getAbsolutePath()) : last;
    }

    private static void moveTmpToFinalWithRetry(File tmp, File finalPath) throws IOException {
        IOException last = null;
        for (int attempt = 0; attempt < FILE_OPERATION_RETRY_COUNT; attempt++) {
            try {
                try {
                    Files.move(tmp.toPath(), finalPath.toPath(),
                            StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE);
                } catch (AtomicMoveNotSupportedException e) {
                    Files.move(tmp.toPath(), finalPath.toPath(), StandardCopyOption.REPLACE_EXISTING);
                }
                return;
            } catch (IOException e) {
                last = e;
                sleepBeforeRetry(attempt, e);
            }
        }
        throw last == null ? new IOException("failed to move tmp raw sample: " + tmp.getAbsolutePath()) : last;
    }

    private static void sleepBeforeRetry(int attempt, IOException cause) throws IOException {
        if (attempt >= FILE_OPERATION_RETRY_COUNT - 1) {
            return;
        }
        try {
            Thread.sleep(FILE_OPERATION_RETRY_BASE_MILLIS * (long) (attempt + 1));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw cause;
        }
    }

    private static void deleteRecursively(File path) {
        if (path == null || !path.exists()) {
            return;
        }
        if (path.isDirectory()) {
            File[] children = path.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        deleteQuietly(path);
    }

    private static String relativeTo(String baseDir, String targetPath) {
        File target = new File(targetPath);
        if (!target.isAbsolute()) {
            return targetPath.replace("\\", "/");
        }
        return new File(baseDir).toPath().relativize(target.getAbsoluteFile().toPath()).toString().replace("\\", "/");
    }

    private static String resolveAgainst(String baseDir, String storedPath) {
        if (storedPath == null || storedPath.isEmpty()) {
            return "";
        }
        File file = new File(storedPath);
        if (file.isAbsolute()) {
            return file.getAbsolutePath();
        }
        return new File(baseDir, storedPath).getAbsolutePath();
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return child == null || child.isNull() ? "" : child.asText();
    }
}
