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
import fs.model.common.StorageType;

import java.io.File;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * array_sample_parquet v1 dataset을 sample별 raw parquet stage와 compact 단계로 만드는 builder.
 *
 * <p>sample 하나가 정상 종료되면 {@code raw_samples/sample_*.parquet}와
 * {@code raw_trace_index/sample_*.parquet}가 먼저 확정된다. 이후 {@link #compact()} 또는
 * {@link #finish()}가 raw 파일들을 size 기반 part로 묶어 최종 {@code sample_parts}와
 * {@code trace_index_parts}를 만든다. 이 구조는 sample 단위 resume과 외부 worker 병렬화에
 * 맞춰져 있다.</p>
 */
/**
 * array_sample_parquet v1 dataset을 sample별 raw parquet stage와 compact 단계로 만드는 builder입니다.
 *
 * <p>sample 하나가 정상 종료되면 `raw_samples/`와 `raw_trace_index/`에 raw parquet가
 * commit됩니다. 마지막에 {@link #finish()} 또는 {@link #compact()}가 raw 파일들을
 * size 기반 part로 묶어 최종 `sample_parts/`와 `trace_index_parts/`를 생성합니다.</p>
 */
public class ArraySampleParquetDatasetBuilder implements AutoCloseable {
    private static final int RAW_STATE_VERSION = 1;
    private static final int RAW_SAMPLE_PADDING = 12;

    private final File outDir;
    private final File rawSamplesDir;
    private final File rawTraceIndexDir;
    private final File samplePartsDir;
    private final File traceIndexPartsDir;
    private final String sampleMetaSourcePath;
    private final String featureMetaSourcePath;
    private final String sampleMetaPath;
    private final String featureMetaPath;
    private final String statePath;
    private final String rawLogPath;
    private final String manifestPath;
    private final ArraySampleParquetBuildOptions options;
    private final List<PointColumnSpec> pointSchema;
    private final int nSamples;
    private final ArrayList<String> sampleKeys;
    private final LinkedHashMap<String, Long> sampleKeyToId;
    private final LinkedHashMap<String, Integer> featureKeyToId = new LinkedHashMap<String, Integer>();
    private final ArrayList<String> featureKeysInOrder = new ArrayList<String>();

    private Integer knownFeatureCount;
    private boolean writesFeatureMeta;
    private boolean finished;
    private boolean closed;
    private Long openSampleId;
    private ArraySampleParquetRawSampleWriter openWriter;
    private ArraySampleParquetFileLock openLock;

    public ArraySampleParquetDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArraySampleParquetBuildOptions options) throws Exception {
        this.outDir = new File(outDir).getAbsoluteFile();
        this.rawSamplesDir = new File(this.outDir, "raw_samples");
        this.rawTraceIndexDir = new File(this.outDir, "raw_trace_index");
        this.samplePartsDir = new File(this.outDir, "sample_parts");
        this.traceIndexPartsDir = new File(this.outDir, "trace_index_parts");
        this.sampleMetaSourcePath = new File(sampleMetaPath).getAbsolutePath();
        this.featureMetaSourcePath = (featureMetaPath == null || featureMetaPath.isEmpty()) ? "" : new File(featureMetaPath).getAbsolutePath();
        this.sampleMetaPath = new File(this.outDir, "sample_meta.parquet").getAbsolutePath();
        this.featureMetaPath = new File(this.outDir, "feature_meta.parquet").getAbsolutePath();
        this.statePath = new File(this.outDir, "raw_state.json").getAbsolutePath();
        this.rawLogPath = new File(this.outDir, "raw_samples.jsonl").getAbsolutePath();
        this.manifestPath = new File(this.outDir, "array_sample_parquet_manifest.json").getAbsolutePath();
        this.options = (options == null) ? new ArraySampleParquetBuildOptions() : options;
        this.pointSchema = normalizeArraySamplePointSchema(pointSchema);

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
            resume(featureMetaPath, featureKeys);
        } else {
            initialize(featureMetaPath, featureKeys);
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

    private void initialize(String featureMetaPath, List<String> featureKeys) throws Exception {
        if (outDir.exists()) {
            String[] children = outDir.list();
            if (children != null && children.length > 0) {
                throw new IllegalArgumentException("out_dir already exists and is not empty: " + outDir.getAbsolutePath());
            }
        }
        ensureDir(rawSamplesDir);
        ensureDir(rawTraceIndexDir);
        ensureDir(samplePartsDir);
        ensureDir(traceIndexPartsDir);
        Files.copy(new File(sampleMetaSourcePath).toPath(), new File(sampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
        initializeFeatures(featureMetaPath, featureKeys);
        cleanupTmpFiles();
        saveState();
    }

    private void initializeFeatures(String featureMetaPath, List<String> featureKeys) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            writesFeatureMeta = false;
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(featureMetaSourcePath);
            knownFeatureCount = Integer.valueOf(featureRows.size());
            Files.copy(new File(featureMetaSourcePath).toPath(), new File(this.featureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            loadFeatureKeys(featureRows);
            return;
        }
        writesFeatureMeta = true;
        if (featureKeys == null) {
            throw new IllegalArgumentException("array_sample_parquet builder requires featureMetaPath or featureKeys");
        }
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
        writeFeatureMetaIfNeeded();
    }

    private void resume(String featureMetaPath, List<String> featureKeys) throws Exception {
        cleanupTmpFiles();
        JsonNode state = JsonUtils.readJson(statePath);
        validateResumeState(state, featureMetaPath, featureKeys);
        finished = state.has("finished") && state.get("finished").asBoolean();
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
    }

    private void validateResumeState(JsonNode state, String featureMetaPath, List<String> featureKeys) throws IOException {
        if (!ArraySampleParquetManifestIO.FORMAT.equals(textOrEmpty(state, "format"))) {
            throw new IOException("unsupported build session format: " + textOrEmpty(state, "format"));
        }
        if (state.has("raw_state_version") && state.get("raw_state_version").asInt() != RAW_STATE_VERSION) {
            throw new IOException("unsupported raw build session version: " + state.get("raw_state_version").asInt());
        }
        if (!new File(textOrEmpty(state, "sample_meta_source_path")).getAbsolutePath().equals(sampleMetaSourcePath)) {
            throw new IOException("sample_meta_path does not match existing build session");
        }
        String expectedFeaturePath = (featureMetaPath == null || featureMetaPath.isEmpty()) ? "" : new File(featureMetaPath).getAbsolutePath();
        if (!textOrEmpty(state, "feature_meta_source_path").equals(expectedFeaturePath)) {
            throw new IOException("feature_meta_path does not match existing build session");
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
        return sample(sampleId, false);
    }

    public ArraySampleParquetSampleContext sample(long sampleId, boolean skipIfCompleted) {
        return new ArraySampleParquetSampleContext(this, sampleId, skipIfCompleted);
    }

    public ArraySampleParquetSampleContext sample(String sampleKey) {
        Long sampleId = sampleKeyToId.get(sampleKey);
        if (sampleId == null) {
            throw new IllegalArgumentException("unknown sample key: " + sampleKey);
        }
        return sample(sampleId.longValue());
    }

    boolean beginSample(long sampleId, boolean skipIfCompleted) {
        ensureOpenForWrites();
        if (openSampleId != null) {
            if (openSampleId.longValue() == sampleId) {
                return false;
            }
            throw new IllegalStateException("another raw sample is already open");
        }
        validateSampleId(sampleId);
        if (isSampleCompleted(sampleId)) {
            if (skipIfCompleted) {
                return true;
            }
            throw new IllegalArgumentException("sample already completed: " + sampleId);
        }

        ArraySampleParquetFileLock lock = new ArraySampleParquetFileLock(rawSamplePath(sampleId) + ".lock");
        try {
            lock.acquire();
            if (isSampleCompleted(sampleId)) {
                if (skipIfCompleted) {
                    lock.release();
                    return true;
                }
                throw new IllegalArgumentException("sample already completed: " + sampleId);
            }
            deleteQuietly(new File(rawSamplePath(sampleId) + ".tmp"));
            deleteQuietly(new File(rawTraceIndexPath(sampleId) + ".tmp"));
            openWriter = new ArraySampleParquetRawSampleWriter(
                    outDir,
                    rawSamplePath(sampleId) + ".tmp",
                    rawTraceIndexPath(sampleId) + ".tmp",
                    pointSchema,
                    options);
            openSampleId = Long.valueOf(sampleId);
            openLock = lock;
            return false;
        } catch (RuntimeException e) {
            lock.release();
            throw e;
        } catch (Exception e) {
            lock.release();
            throw new IllegalStateException("failed to begin raw sample " + sampleId, e);
        }
    }

    void endSample(boolean abort) throws Exception {
        if (openSampleId == null) {
            return;
        }
        long sampleId = openSampleId.longValue();
        ArraySampleParquetRawSampleWriter writer = openWriter;
        ArraySampleParquetFileLock lock = openLock;
        openSampleId = null;
        openWriter = null;
        openLock = null;

        try {
            if (abort) {
                if (writer != null) {
                    writer.abort();
                }
                return;
            }
            if (writer == null) {
                return;
            }
            writer.close();
            File finalPoint = new File(rawSamplePath(sampleId));
            File finalTrace = new File(rawTraceIndexPath(sampleId));
            moveTmpToFinal(new File(rawSamplePath(sampleId) + ".tmp"), finalPoint);
            moveTmpToFinal(new File(rawTraceIndexPath(sampleId) + ".tmp"), finalTrace);
            appendRawCommit(new ArraySampleParquetCompactor.RawSampleRecord(
                    sampleId,
                    sampleKeyForId(Long.valueOf(sampleId)),
                    finalPoint.getAbsolutePath(),
                    finalTrace.getAbsolutePath(),
                    writer.traceCount(),
                    writer.pointCount(),
                    finalPoint.length(),
                    finalTrace.length()));
        } finally {
            if (lock != null) {
                lock.release();
            }
        }
    }

    public void addTrace(long sampleId, Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        ensureOpenForWrites();
        if (openSampleId == null) {
            beginSample(sampleId, false);
        } else if (openSampleId.longValue() != sampleId) {
            throw new IllegalStateException("sample boundary crossed without closing previous raw sample");
        }
        if (openWriter == null) {
            throw new IllegalStateException("raw sample writer is not open");
        }
        int resolvedFeatureId = resolveFeatureId(featureId, featureKey);
        NormalizedColumns normalized = normalizeColumns(columns);
        openWriter.writeTrace(sampleId, resolvedFeatureId, normalized.traceLen, normalized.columns);
    }

    public boolean isSampleCompleted(long sampleId) {
        return rawCommitRecords().containsKey(Long.valueOf(sampleId));
    }

    public List<Long> completedSampleIds() {
        ArrayList<Long> out = new ArrayList<Long>(rawCommitRecords().keySet());
        Collections.sort(out);
        return out;
    }

    public List<Long> pendingSampleIds() {
        LinkedHashSet<Long> completed = new LinkedHashSet<Long>(completedSampleIds());
        ArrayList<Long> out = new ArrayList<Long>();
        for (long sampleId = 0L; sampleId < nSamples; sampleId++) {
            if (!completed.contains(Long.valueOf(sampleId))) {
                out.add(Long.valueOf(sampleId));
            }
        }
        return out;
    }

    public int recoverRawSamples() throws Exception {
        Map<Long, ArraySampleParquetCompactor.RawSampleRecord> known = rawCommitRecords();
        int recovered = 0;
        File[] files = rawSamplesDir.listFiles();
        if (files == null) {
            return 0;
        }
        ArrayList<File> sorted = new ArrayList<File>();
        Collections.addAll(sorted, files);
        Collections.sort(sorted, new Comparator<File>() {
            @Override
            public int compare(File a, File b) {
                return a.getName().compareTo(b.getName());
            }
        });
        for (File pointFile : sorted) {
            Long sampleId = sampleIdFromRawFilename(pointFile.getName());
            if (sampleId == null || sampleId.longValue() < 0L || sampleId.longValue() >= nSamples || known.containsKey(sampleId)) {
                continue;
            }
            File traceFile = new File(rawTraceIndexPath(sampleId.longValue()));
            if (!traceFile.exists()) {
                continue;
            }
            CountPair counts = readRawCounts(pointFile, traceFile, sampleId.longValue());
            if (counts == null) {
                continue;
            }
            appendRawCommit(new ArraySampleParquetCompactor.RawSampleRecord(
                    sampleId.longValue(),
                    sampleKeyForId(sampleId),
                    pointFile.getAbsolutePath(),
                    traceFile.getAbsolutePath(),
                    counts.traceCount,
                    counts.rowCount,
                    pointFile.length(),
                    traceFile.length()));
            known.put(sampleId, null);
            recovered++;
        }
        return recovered;
    }

    public ArraySampleParquetBuildSessionStatus status() {
        List<Long> completed = completedSampleIds();
        List<Long> pending = pendingSampleIds();
        return new ArraySampleParquetBuildSessionStatus(
                nSamples,
                completed,
                pending,
                finished,
                finished ? manifestPath : null,
                openSampleId,
                sampleKeyForId(openSampleId));
    }

    public String compact() throws Exception {
        return compact(true, false, false);
    }

    public String compact(boolean requireAll, boolean cleanupRaw, boolean overwrite) throws Exception {
        if (finished && new File(manifestPath).exists() && !overwrite) {
            return manifestPath;
        }
        ensureOpenForWrites();
        endSample(false);
        writeFeatureMetaIfNeeded();

        Map<Long, ArraySampleParquetCompactor.RawSampleRecord> recordsBySample = rawCommitRecords();
        List<Long> pending = pendingSampleIds();
        if (requireAll && !pending.isEmpty()) {
            throw new IllegalStateException("cannot compact: " + pending.size() + " samples are still pending");
        }
        if (new File(manifestPath).exists() && !overwrite) {
            throw new IllegalStateException("manifest already exists: " + manifestPath);
        }
        ArrayList<ArraySampleParquetCompactor.RawSampleRecord> records = new ArrayList<ArraySampleParquetCompactor.RawSampleRecord>();
        ArrayList<Long> sampleIds = new ArrayList<Long>(recordsBySample.keySet());
        Collections.sort(sampleIds);
        for (Long sampleId : sampleIds) {
            records.add(recordsBySample.get(sampleId));
        }
        List<ArraySampleParquetPart> parts = ArraySampleParquetCompactor.compact(
                outDir,
                samplePartsDir,
                traceIndexPartsDir,
                records,
                pointSchema,
                options,
                overwrite);
        ArraySampleParquetManifest manifest = new ArraySampleParquetManifest(
                1,
                sampleMetaPath,
                featureMetaPath,
                nSamples,
                knownFeatureCount == null ? featureKeysInOrder.size() : knownFeatureCount.intValue(),
                samplePartsDir.getAbsolutePath(),
                traceIndexPartsDir.getAbsolutePath(),
                options.sampleKeyCol,
                options.featureKeyCol,
                pointSchema,
                parts);
        ArraySampleParquetManifestIO.write(manifest, manifestPath);
        finished = true;
        saveState();
        if (cleanupRaw) {
            deleteRecursively(rawSamplesDir);
            deleteRecursively(rawTraceIndexDir);
        }
        return manifestPath;
    }

    public String finish() throws Exception {
        return compact(true, false, false);
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        endSample(false);
        saveState();
        closed = true;
    }

    private void appendRawCommit(ArraySampleParquetCompactor.RawSampleRecord record) throws IOException {
        ArraySampleParquetFileLock lock = new ArraySampleParquetFileLock(rawLogPath + ".lock");
        lock.acquire();
        try {
            ObjectNode node = JsonUtils.objectNode();
            node.put("sample_id", record.sampleId);
            if (record.sampleKey == null) {
                node.putNull("sample_key");
            } else {
                node.put("sample_key", record.sampleKey);
            }
            node.put("path", relativeToOutDir(record.pointPath));
            node.put("trace_index_path", relativeToOutDir(record.traceIndexPath));
            node.put("trace_count", record.traceCount);
            node.put("row_count", record.rowCount);
            node.put("byte_size", record.byteSize);
            node.put("trace_index_byte_size", record.traceIndexByteSize);
            JsonUtils.appendJsonLine(rawLogPath, node);
        } finally {
            lock.release();
        }
    }

    private Map<Long, ArraySampleParquetCompactor.RawSampleRecord> rawCommitRecords() {
        LinkedHashMap<Long, ArraySampleParquetCompactor.RawSampleRecord> records = new LinkedHashMap<Long, ArraySampleParquetCompactor.RawSampleRecord>();
        List<JsonNode> lines;
        try {
            lines = JsonUtils.readJsonLines(rawLogPath);
        } catch (IOException e) {
            throw new IllegalStateException("failed to read raw sample log: " + rawLogPath, e);
        }
        for (JsonNode item : lines) {
            if (!item.has("sample_id")) {
                continue;
            }
            long sampleId = item.get("sample_id").asLong();
            if (sampleId < 0L || sampleId >= nSamples) {
                continue;
            }
            String pointPath = resolveAgainstOutDir(textOrEmpty(item, "path"));
            String tracePath = resolveAgainstOutDir(textOrEmpty(item, "trace_index_path"));
            if (!new File(pointPath).exists() || !new File(tracePath).exists()) {
                continue;
            }
            records.put(Long.valueOf(sampleId), new ArraySampleParquetCompactor.RawSampleRecord(
                    sampleId,
                    item.has("sample_key") && !item.get("sample_key").isNull() ? item.get("sample_key").asText() : sampleKeyForId(Long.valueOf(sampleId)),
                    pointPath,
                    tracePath,
                    item.has("trace_count") ? item.get("trace_count").asInt() : 0,
                    item.has("row_count") ? item.get("row_count").asInt() : 0,
                    item.has("byte_size") ? item.get("byte_size").asLong() : new File(pointPath).length(),
                    item.has("trace_index_byte_size") ? item.get("trace_index_byte_size").asLong() : new File(tracePath).length()));
        }
        return records;
    }

    private CountPair readRawCounts(File pointFile, File traceFile, long sampleId) {
        try (Connection conn = fs.io.common.DuckDBUtils.connect(null);
             Statement st = conn.createStatement()) {
            ArraySampleParquetDuckDB.configure(conn, outDir, options);
            int rowCount;
            int traceCount;
            try (ResultSet rs = st.executeQuery("SELECT count(*) AS n FROM read_parquet(" + fs.io.common.DuckDBUtils.quotePath(pointFile.getAbsolutePath()) + ") WHERE sample_id <> " + sampleId)) {
                rs.next();
                if (rs.getLong("n") != 0L) {
                    return null;
                }
            }
            try (ResultSet rs = st.executeQuery("SELECT count(*) AS n FROM read_parquet(" + fs.io.common.DuckDBUtils.quotePath(traceFile.getAbsolutePath()) + ") WHERE sample_id <> " + sampleId)) {
                rs.next();
                if (rs.getLong("n") != 0L) {
                    return null;
                }
            }
            try (ResultSet rs = st.executeQuery("SELECT count(*) AS n FROM read_parquet(" + fs.io.common.DuckDBUtils.quotePath(pointFile.getAbsolutePath()) + ")")) {
                rs.next();
                rowCount = (int) rs.getLong("n");
            }
            try (ResultSet rs = st.executeQuery("SELECT count(*) AS n FROM read_parquet(" + fs.io.common.DuckDBUtils.quotePath(traceFile.getAbsolutePath()) + ")")) {
                rs.next();
                traceCount = (int) rs.getLong("n");
            }
            return new CountPair(rowCount, traceCount);
        } catch (Exception e) {
            return null;
        }
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
            if (knownFeatureCount != null && id >= knownFeatureCount.intValue()) {
                throw new IllegalArgumentException("featureId out of range: " + id);
            }
            return id;
        }
        Integer resolved = featureKeyToId.get(featureKey);
        if (resolved == null) {
            throw new IllegalArgumentException("unknown feature key: " + featureKey);
        }
        return resolved.intValue();
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
            Object normalized;
            switch (spec.storageType) {
                case FLOAT64:
                    normalized = ArrayUtils.toDoubleArray(value, spec.name);
                    break;
                case INT32:
                    normalized = ArrayUtils.toIntArray(value, spec.name);
                    break;
                case STRING:
                    normalized = ArrayUtils.toStringArray(value, spec.name);
                    break;
                case INT64:
                case UINT8:
                case UINT16:
                case UINT32:
                case UINT64:
                    normalized = ArrayUtils.toLongArray(value, spec.name);
                    break;
                default:
                    throw new IllegalArgumentException("unsupported storage type: " + spec.storageType.value);
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

    private void writeFeatureMetaIfNeeded() throws Exception {
        if (!writesFeatureMeta || new File(featureMetaPath).exists()) {
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

    private void saveState() throws IOException {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format", ArraySampleParquetManifestIO.FORMAT);
        root.put("raw_state_version", RAW_STATE_VERSION);
        root.put("sample_meta_source_path", sampleMetaSourcePath);
        root.put("feature_meta_source_path", featureMetaSourcePath);
        root.put("writes_feature_meta", writesFeatureMeta);
        if (knownFeatureCount == null) {
            root.putNull("known_feature_count");
        } else {
            root.put("known_feature_count", knownFeatureCount.intValue());
        }
        ArrayNode schema = root.putArray("point_schema");
        for (PointColumnSpec spec : pointSchema) {
            ObjectNode item = schema.addObject();
            item.put("name", spec.name);
            item.put("storage_type", spec.storageType.value);
            item.put("logical_type", spec.logicalType.value);
        }
        ArrayNode featureKeys = root.putArray("feature_keys_in_order");
        for (String key : featureKeysInOrder) {
            featureKeys.add(key);
        }
        root.put("finished", finished);
        if (finished) {
            root.put("manifest_path", manifestPath);
        } else {
            root.putNull("manifest_path");
        }
        JsonUtils.writeJsonAtomic(statePath, root);
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

    private static List<PointColumnSpec> normalizeArraySamplePointSchema(List<PointColumnSpec> pointSchema) {
        List<PointColumnSpec> normalized = PointColumnSpec.normalizeList(pointSchema);
        ArrayList<PointColumnSpec> out = new ArrayList<PointColumnSpec>(normalized.size());
        for (PointColumnSpec spec : normalized) {
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                out.add(new PointColumnSpec(spec.name, StorageType.STRING, spec.logicalType));
            } else {
                out.add(new PointColumnSpec(spec.name, spec.storageType, spec.logicalType));
            }
        }
        return out;
    }

    private void ensureOpenForWrites() {
        if (closed) {
            throw new IllegalStateException("builder is closed");
        }
        if (finished) {
            throw new IllegalStateException("builder is already compacted");
        }
    }

    private void validateSampleId(long sampleId) {
        if (sampleId < 0L || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
    }

    private String rawSamplePath(long sampleId) {
        return new File(rawSamplesDir, String.format("sample_%0" + RAW_SAMPLE_PADDING + "d.parquet", sampleId)).getAbsolutePath();
    }

    private String rawTraceIndexPath(long sampleId) {
        return new File(rawTraceIndexDir, String.format("sample_%0" + RAW_SAMPLE_PADDING + "d.parquet", sampleId)).getAbsolutePath();
    }

    private String sampleKeyForId(Long sampleId) {
        if (sampleId == null || sampleId.longValue() < 0L || sampleId.longValue() >= sampleKeys.size()) {
            return null;
        }
        return sampleKeys.get((int) sampleId.longValue());
    }

    private String relativeToOutDir(String path) {
        return outDir.toPath().relativize(new File(path).getAbsoluteFile().toPath()).toString().replace("\\", "/");
    }

    private String resolveAgainstOutDir(String path) {
        if (path == null || path.isEmpty()) {
            return "";
        }
        File file = new File(path);
        if (file.isAbsolute()) {
            return file.getAbsolutePath();
        }
        return new File(outDir, path).getAbsolutePath();
    }

    private Long sampleIdFromRawFilename(String name) {
        if (!name.startsWith("sample_") || !name.endsWith(".parquet")) {
            return null;
        }
        try {
            return Long.valueOf(name.substring("sample_".length(), name.length() - ".parquet".length()));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private void cleanupTmpFiles() {
        cleanupTmpFiles(outDir);
        cleanupTmpFiles(rawSamplesDir);
        cleanupTmpFiles(rawTraceIndexDir);
        cleanupTmpFiles(samplePartsDir);
        cleanupTmpFiles(traceIndexPartsDir);
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
                deleteQuietly(child);
            }
        }
    }

    private static void ensureDir(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("failed to create dir: " + dir.getAbsolutePath());
        }
    }

    private static void moveTmpToFinal(File tmp, File finalPath) throws IOException {
        try {
            Files.move(tmp.toPath(), finalPath.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp.toPath(), finalPath.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void deleteQuietly(File file) {
        if (file.exists() && !file.delete()) {
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
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static ArrayList<String> loadKeys(List<LinkedHashMap<String, Object>> rows, String keyCol) {
        ArrayList<String> out = new ArrayList<String>(rows.size());
        for (Map<String, Object> row : rows) {
            Object value = row.get(keyCol);
            out.add(value == null ? null : value.toString());
        }
        return out;
    }

    private static final class NormalizedColumns {
        final int traceLen;
        final LinkedHashMap<String, Object> columns;

        NormalizedColumns(int traceLen, LinkedHashMap<String, Object> columns) {
            this.traceLen = traceLen;
            this.columns = columns;
        }
    }

    private static final class CountPair {
        final int rowCount;
        final int traceCount;

        CountPair(int rowCount, int traceCount) {
            this.rowCount = rowCount;
            this.traceCount = traceCount;
        }
    }
}
