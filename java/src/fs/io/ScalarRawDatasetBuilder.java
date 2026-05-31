package fs.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.config.BuildShardConfig;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.JsonUtils;
import fs.io.scalar.ScalarDenseLongShardBuilder;
import fs.io.scalar.ScalarMetadataWriter;
import fs.io.scalar.ScalarRawSampleWriter;
import fs.io.scalar.ScalarSampleBundleManifestIO;
import fs.io.scalar.ShardBuilder;
import fs.model.scalar.ScalarRawBuildStatus;
import fs.model.scalar.ScalarSampleBundleManifest;

import java.io.File;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * scalar 값을 sample별 raw parquet 파일로 먼저 저장하는 random-order builder다.
 *
 * <p>기존 {@link ScalarDatasetBuilder}는 sample을 순서대로 받아 내부 bundle을 만든다.
 * 이 class는 Python {@code ScalarRawDatasetBuilder}와 같은 모델로, sample 하나를
 * {@code raw_samples/sample_*.parquet} 파일 하나에 commit한다. 따라서 외부 supervisor가
 * {@link #status()}의 pending sample 목록을 worker들에게 임의 순서로 나눠줄 수 있다.
 */
public class ScalarRawDatasetBuilder implements AutoCloseable {
    private static final int RAW_STATE_VERSION = 1;
    private static final int RAW_SAMPLE_PADDING = 12;

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

    public ScalarRawDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
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

    public static ScalarRawDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig) throws Exception {
        return new ScalarRawDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig);
    }

    public ScalarRawBuildStatus status() throws Exception {
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
        return new ScalarRawBuildStatus(nSamples, completed, pending, finishedStage, finishedStage ? sampleMajorManifestPath : "");
    }

    public void writeSample(long sampleId, Map<?, ?> values) throws Exception {
        writeSample(sampleId, values, false);
    }

    public void writeSample(long sampleId, Map<?, ?> values, boolean skipIfCompleted) throws Exception {
        ensureOpenForWrites();
        validateSampleId(sampleId);
        if (isSampleCompleted(sampleId)) {
            if (skipIfCompleted) {
                return;
            }
            throw new IllegalArgumentException("sample already completed: " + sampleId);
        }

        File lockFile = new File(rawSamplePath(sampleId) + ".lock");
        FileChannel lockChannel = FileChannel.open(
                lockFile.toPath(),
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.WRITE);
        try (FileChannel channel = lockChannel; FileLock ignored = channel.lock()) {
            if (isSampleCompleted(sampleId)) {
                if (skipIfCompleted) {
                    return;
                }
                throw new IllegalArgumentException("sample already completed: " + sampleId);
            }
            LinkedHashMap<Integer, Double> normalized = normalizeValues(values);
            String finalPath = rawSamplePath(sampleId);
            String tmpPath = finalPath + ".tmp";
            deleteQuietly(new File(tmpPath));
            int rowCount = ScalarRawSampleWriter.write(tmpPath, sampleId, normalized);
            Files.move(new File(tmpPath).toPath(), new File(finalPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            appendCommit(sampleId, finalPath, rowCount);
            saveState();
        }
    }

    public String finishStage() throws Exception {
        return finishStage(true);
    }

    public String finishStage(boolean requireAll) throws Exception {
        ensureOpen();
        if (finishedStage) {
            return sampleMajorManifestPath;
        }
        if (requireAll && !status().pendingSampleIds.isEmpty()) {
            throw new IllegalStateException("cannot finish scalar raw stage: pending sample count=" + status().pendingSampleIds.size());
        }
        List<Long> completed = completedSampleIds();
        Collections.sort(completed);
        ArrayList<String> paths = new ArrayList<String>(completed.size());
        for (Long sampleId : completed) {
            paths.add(rawSamplePath(sampleId.longValue()));
        }
        ScalarSampleBundleManifestIO.write(
                new ScalarSampleBundleManifest(
                        sampleMetaPath,
                        featureMetaPath,
                        paths,
                        buildConfig.sampleIdCol,
                        buildConfig.featureIdCol,
                        buildConfig.valueCol),
                sampleMajorManifestPath);
        finishedStage = true;
        saveState();
        return sampleMajorManifestPath;
    }

    public String buildBlobShards(boolean requireAll, boolean keepRaw) throws Exception {
        String stageManifest = finishStage(requireAll);
        String manifest = ShardBuilder.buildShardsFromSampleBundles(stageManifest, outDir.getAbsolutePath(), buildConfig);
        if (!keepRaw) {
            deleteRecursively(rawSamplesDir);
        }
        return manifest;
    }

    public String buildDenseLongShards(boolean requireAll, String denseLongOutDir) throws Exception {
        String stageManifest = finishStage(requireAll);
        String target = denseLongOutDir == null || denseLongOutDir.isEmpty()
                ? new File(outDir, "dense_long_shards").getAbsolutePath()
                : denseLongOutDir;
        return ScalarDenseLongShardBuilder.buildFromSampleBundles(stageManifest, target, buildConfig);
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
        if (featureKeys == null) {
            throw new IllegalArgumentException("ScalarRawDatasetBuilder requires featureMetaPath or featureKeys");
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
            throw new IOException("unsupported scalar raw stage format: " + textOrEmpty(state, "format"));
        }
        if (!new File(textOrEmpty(state, "sample_meta_source_path")).getAbsolutePath().equals(sampleMetaSourcePath)) {
            throw new IOException("sampleMetaPath does not match existing scalar raw stage");
        }
        String expectedFeature = featureMetaPath == null || featureMetaPath.isEmpty() ? "" : new File(featureMetaPath).getAbsolutePath();
        if (!textOrEmpty(state, "feature_meta_source_path").equals(expectedFeature)) {
            throw new IOException("featureMetaPath does not match existing scalar raw stage");
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
            throw new IOException("featureKeys do not match existing scalar raw stage");
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
        File lockPath = new File(rawLogPath + ".lock");
        File parent = lockPath.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileChannel channel = FileChannel.open(
                lockPath.toPath(),
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.WRITE);
             FileLock ignored = channel.lock()) {
            JsonUtils.appendJsonLine(rawLogPath, node);
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
            throw new IllegalStateException("scalar raw builder is closed");
        }
    }

    private void ensureOpenForWrites() {
        ensureOpen();
        if (finishedStage) {
            throw new IllegalStateException("scalar raw stage has already been finalized");
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
        if (path != null && path.exists() && !path.delete()) {
            // best-effort cleanup
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            // best-effort cleanup
        }
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
