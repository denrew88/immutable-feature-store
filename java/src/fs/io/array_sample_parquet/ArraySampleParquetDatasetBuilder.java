package fs.io.array_sample_parquet;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.ArrayUtils;
import fs.io.common.JsonUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetBuildSessionStatus;
import fs.model.array_sample_parquet.ArraySampleParquetManifest;
import fs.model.array_sample_parquet.ArraySampleParquetPart;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * array_sample_parquet v1 dataset을 sample 순서대로 생성하는 resume-safe builder다.
 *
 * <p>사용자는 sample 단위로 trace를 넣는다. builder는 자동으로 sample들을 part
 * parquet로 끊고, part commit 후 `parts.jsonl`과 `state.json`을 갱신한다.
 * 중간에 프로세스가 죽으면 `status().nextExpectedSampleId`부터 다시 넣으면 된다.
 */
public class ArraySampleParquetDatasetBuilder implements AutoCloseable {
    private final File outDir;
    private final File samplePartsDir;
    private final File dictionaryDir;
    private final String sampleMetaSourcePath;
    private final String featureMetaSourcePath;
    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final String statePath;
    private final String partsLogPath;
    private final String manifestPath;
    private final ArraySampleParquetBuildOptions options;
    private final List<PointColumnSpec> pointSchema;
    private final int nSamples;
    private final ArrayList<String> sampleKeys;
    private final LinkedHashMap<String, Long> sampleKeyToId;
    private final LinkedHashMap<String, Integer> featureKeyToId = new LinkedHashMap<String, Integer>();
    private final ArrayList<String> featureKeysInOrder = new ArrayList<String>();
    private final LinkedHashMap<String, CategoricalRegistry> categoricalRegistries = new LinkedHashMap<String, CategoricalRegistry>();
    private boolean knownFeatureMode;
    private boolean writesFeatureMeta;
    private Integer knownFeatureCount;
    private int committedPartCount;
    private Long lastCommittedSampleId;
    private long cursorSampleId;
    private Long openSampleId;
    private int currentSampleTraceCount;
    private Long pendingFirstSampleId;
    private Long pendingLastSampleId;
    private int pendingSampleCount;
    private int pendingTraceCount;
    private int partRows;
    private long partBytes;
    private ArraySampleParquetPartWriter partWriter;
    private boolean finished;
    private boolean closed;

    public ArraySampleParquetDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArraySampleParquetBuildOptions options) throws Exception {
        this.outDir = new File(outDir).getAbsoluteFile();
        this.samplePartsDir = new File(this.outDir, "sample_parts");
        this.dictionaryDir = new File(this.outDir, "categorical_dictionaries");
        this.sampleMetaSourcePath = new File(sampleMetaPath).getAbsolutePath();
        this.featureMetaSourcePath = (featureMetaPath == null || featureMetaPath.isEmpty()) ? "" : new File(featureMetaPath).getAbsolutePath();
        this.sampleMetaPath = new File(this.outDir, "sample_meta.parquet").getAbsolutePath();
        this.featureMetaPath = new File(this.outDir, "feature_meta.parquet").getAbsolutePath();
        this.statePath = new File(this.outDir, "state.json").getAbsolutePath();
        this.partsLogPath = new File(this.outDir, "parts.jsonl").getAbsolutePath();
        this.manifestPath = new File(this.outDir, "array_sample_parquet_manifest.json").getAbsolutePath();
        this.options = (options == null) ? new ArraySampleParquetBuildOptions() : options;
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        for (PointColumnSpec spec : this.pointSchema) {
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                categoricalRegistries.put(spec.name, new CategoricalRegistry());
            }
        }

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sampleMetaSourcePath);
        this.nSamples = sampleRows.size();
        this.sampleKeys = loadKeys(sampleRows, this.options.sampleKeyCol);
        this.sampleKeyToId = new LinkedHashMap<String, Long>();
        for (int i = 0; i < sampleKeys.size(); i++) {
            String key = sampleKeys.get(i);
            if (key != null) {
                sampleKeyToId.put(key, Long.valueOf(i));
            }
        }

        if (new File(statePath).exists()) {
            resume(featureKeys);
        } else {
            initialize(featureKeys);
        }
    }

    public static ArraySampleParquetDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArraySampleParquetBuildOptions options) throws Exception {
        return new ArraySampleParquetDatasetBuilder(outDir, sampleMetaPath, pointSchema, featureMetaPath, featureKeys, options);
    }

    private void initialize(List<String> featureKeys) throws Exception {
        if (outDir.exists()) {
            String[] children = outDir.list();
            if (children != null && children.length > 0) {
                throw new IllegalArgumentException("out_dir already exists and is not empty: " + outDir.getAbsolutePath());
            }
        }
        ensureDir(samplePartsDir);
        Files.copy(new File(sampleMetaSourcePath).toPath(), new File(sampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
        initializeFeatures(featureKeys);
        saveState();
    }

    private void initializeFeatures(List<String> featureKeys) throws Exception {
        if (!featureMetaSourcePath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        if (!featureMetaSourcePath.isEmpty()) {
            knownFeatureMode = true;
            writesFeatureMeta = false;
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(featureMetaSourcePath);
            knownFeatureCount = Integer.valueOf(featureRows.size());
            Files.copy(new File(featureMetaSourcePath).toPath(), new File(featureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            loadFeatureKeys(featureRows);
            return;
        }
        writesFeatureMeta = true;
        if (featureKeys != null) {
            knownFeatureMode = true;
            LinkedHashSet<String> seen = new LinkedHashSet<String>();
            for (String key : featureKeys) {
                String text = String.valueOf(key);
                if (!seen.add(text)) {
                    throw new IllegalArgumentException("duplicate feature key: " + text);
                }
                featureKeyToId.put(text, Integer.valueOf(featureKeysInOrder.size()));
                featureKeysInOrder.add(text);
            }
            knownFeatureCount = Integer.valueOf(featureKeysInOrder.size());
        }
    }

    private void resume(List<String> featureKeys) throws Exception {
        cleanupTmpFiles(outDir);
        cleanupTmpFiles(samplePartsDir);
        cleanupTmpFiles(dictionaryDir);
        JsonNode state = JsonUtils.readJson(statePath);
        validateResumeState(state, featureKeys);
        List<JsonNode> parts = JsonUtils.readJsonLines(partsLogPath);
        committedPartCount = parts.size();
        if (!parts.isEmpty()) {
            lastCommittedSampleId = Long.valueOf(parts.get(parts.size() - 1).get("last_sample_id").asLong());
        }
        cursorSampleId = resumeNextSampleId();
        finished = state.has("finished") && state.get("finished").asBoolean();
        knownFeatureMode = state.has("known_feature_mode") && state.get("known_feature_mode").asBoolean();
        writesFeatureMeta = state.has("writes_feature_meta") && state.get("writes_feature_meta").asBoolean();
        if (state.has("known_feature_count") && !state.get("known_feature_count").isNull()) {
            knownFeatureCount = Integer.valueOf(state.get("known_feature_count").asInt());
        }
        JsonNode featureKeysNode = state.get("feature_keys_in_order");
        if (featureKeysNode != null && featureKeysNode.isArray()) {
            for (JsonNode item : featureKeysNode) {
                String key = item.asText();
                featureKeyToId.put(key, Integer.valueOf(featureKeysInOrder.size()));
                featureKeysInOrder.add(key);
            }
        }
        JsonNode labelsNode = state.get("categorical_labels_by_column");
        if (!parts.isEmpty() && parts.get(parts.size() - 1).has("categorical_labels_by_column")) {
            labelsNode = parts.get(parts.size() - 1).get("categorical_labels_by_column");
        }
        loadCategoricalLabels(labelsNode);
    }

    private void validateResumeState(JsonNode state, List<String> featureKeys) throws IOException {
        if (!"array-sample-parquet".equals(textOrEmpty(state, "format"))) {
            throw new IOException("unsupported build session format: " + textOrEmpty(state, "format"));
        }
        if (!new File(textOrEmpty(state, "sample_meta_source_path")).getAbsolutePath().equals(sampleMetaSourcePath)) {
            throw new IOException("sample_meta_path does not match existing build session");
        }
        if (featureKeys != null) {
            JsonNode stored = state.get("feature_keys_in_order");
            if (stored == null || !stored.isArray() || stored.size() != featureKeys.size()) {
                throw new IOException("feature_keys do not match existing build session");
            }
            for (int i = 0; i < featureKeys.size(); i++) {
                if (!String.valueOf(featureKeys.get(i)).equals(stored.get(i).asText())) {
                    throw new IOException("feature_keys do not match existing build session");
                }
            }
        }
    }

    public ArraySampleParquetSampleContext sample(long sampleId) {
        return new ArraySampleParquetSampleContext(this, sampleId);
    }

    public ArraySampleParquetSampleContext sample(String sampleKey) {
        Long sampleId = sampleKeyToId.get(sampleKey);
        if (sampleId == null) {
            throw new IllegalArgumentException("unknown sample key: " + sampleKey);
        }
        return sample(sampleId.longValue());
    }

    void beginSample(long sampleId) {
        ensureOpenForWrites();
        if (openSampleId != null) {
            if (openSampleId.longValue() == sampleId) {
                return;
            }
            throw new IllegalStateException("another sample is already open");
        }
        if (sampleId != cursorSampleId) {
            throw new IllegalArgumentException("array_sample_parquet session expects sample_id " + cursorSampleId + "; got " + sampleId);
        }
        openSampleId = Long.valueOf(sampleId);
        currentSampleTraceCount = 0;
    }

    void endSample(boolean abort) throws Exception {
        if (openSampleId == null) {
            return;
        }
        long sampleId = openSampleId.longValue();
        int traceCount = currentSampleTraceCount;
        openSampleId = null;
        currentSampleTraceCount = 0;
        if (abort) {
            discardPartBuffer();
            return;
        }
        if (pendingFirstSampleId == null) {
            pendingFirstSampleId = Long.valueOf(sampleId);
        }
        pendingLastSampleId = Long.valueOf(sampleId);
        pendingSampleCount++;
        pendingTraceCount += traceCount;
        cursorSampleId = sampleId + 1L;
        if (shouldFlushPart()) {
            flushPart();
        }
    }

    public void addTrace(long sampleId, Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        ensureOpenForWrites();
        if (openSampleId == null) {
            beginSample(sampleId);
        } else if (openSampleId.longValue() != sampleId) {
            throw new IllegalStateException("sample boundary crossed without closing previous sample");
        }
        int resolvedFeatureId = resolveFeatureId(featureId, featureKey);
        NormalizedColumns normalized = normalizeColumns(columns);
        ensurePartWriter().writeRow(sampleId, resolvedFeatureId, normalized.traceLen, normalized.columns);
        partRows++;
        partBytes += estimateRowBytes(normalized.traceLen);
        currentSampleTraceCount++;
    }

    private int resolveFeatureId(Integer featureId, String featureKey) {
        if (featureId == null && featureKey == null) {
            throw new IllegalArgumentException("provide either featureId or featureKey");
        }
        if (featureId != null && featureKey != null) {
            int resolved = resolveFeatureId(null, featureKey);
            if (featureId.intValue() != resolved) {
                throw new IllegalArgumentException("featureId/featureKey mismatch");
            }
            return featureId.intValue();
        }
        if (featureId != null) {
            int id = featureId.intValue();
            if (id < 0) {
                throw new IllegalArgumentException("featureId must be >= 0");
            }
            if (!knownFeatureMode) {
                throw new IllegalArgumentException("featureId inputs require feature metadata or featureKeys");
            }
            if (knownFeatureCount != null && id >= knownFeatureCount.intValue()) {
                throw new IllegalArgumentException("featureId out of range: " + id);
            }
            return id;
        }
        Integer resolved = featureKeyToId.get(featureKey);
        if (resolved != null) {
            return resolved.intValue();
        }
        if (knownFeatureMode) {
            throw new IllegalArgumentException("unknown feature key: " + featureKey);
        }
        int id = featureKeysInOrder.size();
        featureKeyToId.put(featureKey, Integer.valueOf(id));
        featureKeysInOrder.add(featureKey);
        return id;
    }

    private NormalizedColumns normalizeColumns(Map<String, Object> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("columns must not be null");
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<String, Object>();
        Integer traceLen = null;
        for (PointColumnSpec spec : pointSchema) {
            if (!columns.containsKey(spec.name)) {
                throw new IllegalArgumentException("missing point column: " + spec.name);
            }
            Object value = columns.get(spec.name);
            Object normalized = value;
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                normalized = categoricalRegistries.get(spec.name).encode(value, spec.name);
            }
            int columnLen = ArrayUtils.pointColumnLength(normalized);
            if (traceLen == null) {
                traceLen = Integer.valueOf(columnLen);
            } else if (traceLen.intValue() != columnLen) {
                throw new IllegalArgumentException("point column length mismatch for " + spec.name);
            }
            out.put(spec.name, normalized);
        }
        if (columns.size() != pointSchema.size()) {
            for (String name : columns.keySet()) {
                if (!out.containsKey(name)) {
                    throw new IllegalArgumentException("unexpected point column: " + name);
                }
            }
        }
        return new NormalizedColumns(traceLen == null ? 0 : traceLen.intValue(), out);
    }

    public ArraySampleParquetBuildSessionStatus status() {
        long next = resumeNextSampleId();
        return new ArraySampleParquetBuildSessionStatus(
                lastCommittedSampleId,
                sampleKeyForId(lastCommittedSampleId),
                next,
                sampleKeyForId(Long.valueOf(next)),
                committedPartCount,
                finished,
                finished ? manifestPath : null,
                pendingLastSampleId,
                sampleKeyForId(pendingLastSampleId),
                openSampleId,
                sampleKeyForId(openSampleId));
    }

    public Map<String, Object> flushPart() throws Exception {
        if (pendingSampleCount <= 0) {
            return null;
        }
        int partId = committedPartCount;
        String partPath = partPath(partId);
        ensurePartWriter().close();
        partWriter = null;
        long byteSize = new File(partPath).length();
        ObjectNode record = JsonUtils.objectNode();
        record.put("part_id", partId);
        record.put("path", new File(outDir, "sample_parts/part_" + String.format("%06d", partId) + ".parquet").toPath()
                .toAbsolutePath().normalize().toString().replace("\\", "/"));
        record.put("first_sample_id", pendingFirstSampleId.longValue());
        record.put("last_sample_id", pendingLastSampleId.longValue());
        record.put("first_sample_key", sampleKeyForId(pendingFirstSampleId));
        record.put("last_sample_key", sampleKeyForId(pendingLastSampleId));
        record.put("sample_count", pendingSampleCount);
        record.put("trace_count", pendingTraceCount);
        record.put("row_count", partRows);
        record.put("byte_size", byteSize);
        writeCategoricalLabels(record.putObject("categorical_labels_by_column"));
        JsonUtils.appendJsonLine(partsLogPath, record);
        committedPartCount++;
        lastCommittedSampleId = pendingLastSampleId;
        resetPartBuffer();
        saveState();
        return null;
    }

    public String finish() throws Exception {
        if (finished) {
            return manifestPath;
        }
        ensureOpenForWrites();
        endSample(false);
        flushPart();
        writeFeatureMetaIfNeeded();
        List<PointColumnSpec> schemaWithDictionaries = writeDictionaries();
        List<ArraySampleParquetPart> parts = readCommittedParts();
        int nFeatures = knownFeatureCount == null ? featureKeysInOrder.size() : knownFeatureCount.intValue();
        ArraySampleParquetManifest manifest = new ArraySampleParquetManifest(
                1,
                sampleMetaPath,
                featureMetaPath,
                nSamples,
                nFeatures,
                samplePartsDir.getAbsolutePath(),
                options.sampleKeyCol,
                options.featureKeyCol,
                schemaWithDictionaries,
                parts);
        ArraySampleParquetManifestIO.write(manifest, manifestPath);
        finished = true;
        saveState();
        return manifestPath;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        if (!finished) {
            endSample(false);
            flushPart();
            saveState();
        }
        closed = true;
    }

    private boolean shouldFlushPart() {
        if (options.maxPartSamples > 0 && pendingSampleCount >= options.maxPartSamples) {
            return true;
        }
        return partRows >= options.maxPartRows
                || partBytes >= options.targetPartBytes;
    }

    private long estimateRowBytes(int traceLen) {
        long bytes = 8L + 4L + 4L;
        for (PointColumnSpec spec : pointSchema) {
            bytes += (long) traceLen * (long) spec.storageType.itemSize;
        }
        return bytes;
    }

    private void writeFeatureMetaIfNeeded() throws Exception {
        if (!writesFeatureMeta) {
            return;
        }
        ArrayList<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        for (int i = 0; i < featureKeysInOrder.size(); i++) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("feature_id", Integer.valueOf(i));
            row.put(options.featureKeyCol, featureKeysInOrder.get(i));
            rows.add(row);
        }
        ArrayMetadataWriter.writeFeatureMeta(rows, featureMetaPath);
        knownFeatureCount = Integer.valueOf(featureKeysInOrder.size());
    }

    private List<PointColumnSpec> writeDictionaries() throws IOException {
        ensureDir(dictionaryDir);
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>();
        for (PointColumnSpec spec : pointSchema) {
            if (spec.logicalType != LogicalType.CATEGORICAL) {
                out.add(spec);
                continue;
            }
            String path = new File(dictionaryDir, spec.name + ".json").getAbsolutePath();
            ArraySampleParquetDictionaryIO.write(path, spec.name, categoricalRegistries.get(spec.name).labels());
            out.add(spec.withDictionaryPath(path));
        }
        return out;
    }

    private List<ArraySampleParquetPart> readCommittedParts() throws IOException {
        ArrayList<ArraySampleParquetPart> out = new ArrayList<ArraySampleParquetPart>();
        List<JsonNode> records = JsonUtils.readJsonLines(partsLogPath);
        for (JsonNode item : records) {
            out.add(new ArraySampleParquetPart(
                    item.get("part_id").asInt(),
                    textOrEmpty(item, "path"),
                    item.get("first_sample_id").asLong(),
                    item.get("last_sample_id").asLong(),
                    item.get("sample_count").asInt(),
                    item.get("trace_count").asInt(),
                    item.get("row_count").asInt(),
                    item.get("byte_size").asLong()));
        }
        return out;
    }

    private void saveState() throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", "array-sample-parquet");
        root.put("state_version", 1);
        root.put("sample_meta_source_path", sampleMetaSourcePath);
        root.put("feature_meta_source_path", featureMetaSourcePath);
        root.put("known_feature_mode", knownFeatureMode);
        root.put("writes_feature_meta", writesFeatureMeta);
        if (knownFeatureCount == null) {
            root.putNull("known_feature_count");
        } else {
            root.put("known_feature_count", knownFeatureCount.intValue());
        }
        ArrayNode featureKeys = root.putArray("feature_keys_in_order");
        for (String key : featureKeysInOrder) {
            featureKeys.add(key);
        }
        writeCategoricalLabels(root.putObject("categorical_labels_by_column"));
        root.put("committed_part_count", committedPartCount);
        if (lastCommittedSampleId == null) {
            root.putNull("last_committed_sample_id");
            root.putNull("last_committed_sample_key");
        } else {
            root.put("last_committed_sample_id", lastCommittedSampleId.longValue());
            root.put("last_committed_sample_key", sampleKeyForId(lastCommittedSampleId));
        }
        long next = resumeNextSampleId();
        root.put("next_expected_sample_id", next);
        String nextKey = sampleKeyForId(Long.valueOf(next));
        if (nextKey == null) {
            root.putNull("next_expected_sample_key");
        } else {
            root.put("next_expected_sample_key", nextKey);
        }
        root.put("finished", finished);
        if (finished) {
            root.put("manifest_path", manifestPath);
        } else {
            root.putNull("manifest_path");
        }
        JsonUtils.writeJsonAtomic(statePath, root);
    }

    private void writeCategoricalLabels(ObjectNode labelsRoot) {
        for (Map.Entry<String, CategoricalRegistry> entry : categoricalRegistries.entrySet()) {
            ArrayNode labels = labelsRoot.putArray(entry.getKey());
            for (String label : entry.getValue().labels()) {
                labels.add(label);
            }
        }
    }

    private void loadCategoricalLabels(JsonNode labelsNode) {
        if (labelsNode == null || !labelsNode.isObject()) {
            return;
        }
        for (PointColumnSpec spec : pointSchema) {
            if (spec.logicalType != LogicalType.CATEGORICAL) {
                continue;
            }
            JsonNode raw = labelsNode.get(spec.name);
            ArrayList<String> labels = new ArrayList<String>();
            if (raw != null && raw.isArray()) {
                for (JsonNode item : raw) {
                    labels.add(item.asText());
                }
            }
            categoricalRegistries.get(spec.name).loadLabels(labels);
        }
    }

    private void resetPartBuffer() {
        partWriter = null;
        partRows = 0;
        partBytes = 0L;
        pendingFirstSampleId = null;
        pendingLastSampleId = null;
        pendingSampleCount = 0;
        pendingTraceCount = 0;
    }

    private void discardPartBuffer() {
        if (partWriter != null) {
            partWriter.abort();
        }
        resetPartBuffer();
        cursorSampleId = resumeNextSampleId();
    }

    private long resumeNextSampleId() {
        return lastCommittedSampleId == null ? 0L : lastCommittedSampleId.longValue() + 1L;
    }

    private String sampleKeyForId(Long sampleId) {
        if (sampleId == null || sampleId.longValue() < 0L || sampleId.longValue() >= sampleKeys.size()) {
            return null;
        }
        return sampleKeys.get((int) sampleId.longValue());
    }

    private String partPath(int partId) {
        return new File(samplePartsDir, String.format("part_%06d.parquet", partId)).getAbsolutePath();
    }

    private ArraySampleParquetPartWriter ensurePartWriter() throws IOException {
        if (partWriter == null) {
            partWriter = ArraySampleParquetPartWriter.open(partPath(committedPartCount), pointSchema, options.compression);
        }
        return partWriter;
    }

    private void ensureOpenForWrites() {
        if (closed) {
            throw new IllegalStateException("builder is closed");
        }
        if (finished) {
            throw new IllegalStateException("builder is already finished");
        }
    }

    private static ArrayList<String> loadKeys(List<LinkedHashMap<String, Object>> rows, String keyCol) {
        ArrayList<String> out = new ArrayList<String>(rows.size());
        for (Map<String, Object> row : rows) {
            Object value = row.get(keyCol);
            out.add(value == null ? null : value.toString());
        }
        return out;
    }

    private void loadFeatureKeys(List<LinkedHashMap<String, Object>> featureRows) {
        for (int i = 0; i < featureRows.size(); i++) {
            Object value = featureRows.get(i).get(options.featureKeyCol);
            if (value == null) {
                continue;
            }
            String key = value.toString();
            featureKeyToId.put(key, Integer.valueOf(i));
            featureKeysInOrder.add(key);
        }
    }

    private static void ensureDir(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("failed to create dir: " + dir.getAbsolutePath());
        }
    }

    private static void cleanupTmpFiles(File dir) {
        if (dir == null || !dir.isDirectory()) {
            return;
        }
        File[] children = dir.listFiles();
        if (children == null) {
            return;
        }
        for (File child : children) {
            if (child.getName().endsWith(".tmp")) {
                child.delete();
            }
        }
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static final class NormalizedColumns {
        final int traceLen;
        final LinkedHashMap<String, Object> columns;

        NormalizedColumns(int traceLen, LinkedHashMap<String, Object> columns) {
            this.traceLen = traceLen;
            this.columns = columns;
        }
    }

}
