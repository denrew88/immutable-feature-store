package fs.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.config.ArrayBinaryBuildOptions;
import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.io.array.ArraySampleBundleWriter;
import fs.io.array.ArrayShardBuilder;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.ArrayUtils;
import fs.io.common.JsonUtils;
import fs.model.array.ArrayBuildSessionStatus;
import fs.model.common.LogicalType;
import fs.model.common.PointColumnSpec;
import fs.model.common.StorageType;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * trace를 sample 단위로 받아 resumable array bundle stage를 만들고,
 * 그 stage를 최종 array binary shard artifact로 변환하는 builder다.
 *
 * <p>자동 chunking은 내부 구현으로 숨겨지고, 커밋 단위는 seal된 bundle parquet
 * 하나다. 사용자는 {@link #status()}를 보고 마지막 committed sample 다음부터
 * 다시 넣으면 된다.
 */
public class ArrayDatasetBuilder implements AutoCloseable {
    private static final int STATE_VERSION = 1;

    private final String outDir;
    private final String sampleMetaPath;
    private final String bundleOutDir;
    private final String bundleSampleMetaPath;
    private final String bundleManifestPath;
    private final String featureMetaPath;
    private final String statePath;
    private final String bundleLogPath;
    private final String categoricalDictDir;
    private final ArrayBinaryBuildOptions buildOptions;
    private final ArrayShardConfig shardConfig;
    private final ArrayBundleConfig bundleConfig;
    private final List<PointColumnSpec> pointSchema;
    private final int nSamples;
    private final ArrayList<String> sampleKeys;
    private final HashMap<String, Long> sampleKeyToId;
    private final HashMap<String, Integer> featureKeyToId;
    private final ArrayList<String> featureKeysInOrder;
    private final HashMap<String, CategoricalRegistry> categoricalRegistries;
    private final boolean knownFeatureMode;
    private final boolean writesFeatureMeta;
    private final Integer knownFeatureCount;
    private final String featureMetaSourcePath;
    private final ArraySampleBundleWriter bundleWriter;

    private boolean closed;
    private boolean bundlesFinalized;
    private boolean finished;
    private String manifestPath;

    private Long lastCommittedSampleId;
    private int committedBundleCount;
    private long cursorSampleId;
    private Long pendingBundleFirstSampleId;
    private Long pendingBundleLastSampleId;
    private int pendingBundleSampleCount;
    private int pendingBundleTraceCount;
    private Long openSampleId;
    private int currentSampleTraceCount;

    /**
     * 기본 설정으로 새 세션을 열거나 기존 세션을 이어받는다.
     */
    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        this(outDir, sampleMetaPath, pointSchema, "", null, null, null, null, "");
    }

    /**
     * build 옵션을 지정해서 세션을 연다.
     */
    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        this(outDir, sampleMetaPath, pointSchema, "", null, buildOptions, null, null, "");
    }

    /**
     * known-feature metadata와 build 옵션을 같이 주고 세션을 연다.
     */
    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        this(outDir, sampleMetaPath, pointSchema, featureMetaPath, null, buildOptions, null, null, "");
    }

    /**
     * shard/bundle config를 직접 주고 세션을 연다.
     */
    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArrayShardConfig shardConfig,
            ArrayBundleConfig bundleConfig,
            String bundleOutDir) throws Exception {
        this(outDir, sampleMetaPath, pointSchema, featureMetaPath, featureKeys, null, shardConfig, bundleConfig, bundleOutDir);
    }

    /**
     * 모든 옵션을 직접 주고 세션을 연다.
     */
    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArrayBinaryBuildOptions buildOptions,
            ArrayShardConfig shardConfig,
            ArrayBundleConfig bundleConfig,
            String bundleOutDir) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        this.outDir = new File(outDir).getAbsolutePath();
        this.sampleMetaPath = new File(sampleMetaPath).getAbsolutePath();
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        this.buildOptions = (buildOptions == null) ? new ArrayBinaryBuildOptions() : buildOptions;
        validateBuildOptions(this.buildOptions);
        this.shardConfig = (shardConfig == null) ? shardConfigFromOptions(this.buildOptions) : shardConfig;
        this.bundleConfig = (bundleConfig == null) ? new ArrayBundleConfig() : bundleConfig;
        this.closed = false;
        this.bundlesFinalized = false;
        this.finished = false;
        this.manifestPath = "";

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sampleMetaPath);
        validateDenseIds(sampleRows, "sample_id", "sample");
        this.nSamples = sampleRows.size();
        this.sampleKeys = new ArrayList<String>(this.nSamples);
        this.sampleKeyToId = new HashMap<String, Long>();
        for (int sampleId = 0; sampleId < sampleRows.size(); sampleId++) {
            Object sampleKey = sampleRows.get(sampleId).get(this.buildOptions.sampleKeyCol);
            String key = (sampleKey == null) ? null : sampleKey.toString();
            sampleKeys.add(key);
            if (key != null) {
                sampleKeyToId.put(key, Long.valueOf(sampleId));
            }
        }

        File bundleRoot = (bundleOutDir == null || bundleOutDir.isEmpty())
                ? new File(this.outDir, "bundle_stage")
                : new File(bundleOutDir);
        this.bundleOutDir = bundleRoot.getAbsolutePath();
        this.bundleSampleMetaPath = new File(bundleRoot, "sample_meta.parquet").getAbsolutePath();
        this.bundleManifestPath = new File(bundleRoot, "array_bundle_manifest.json").getAbsolutePath();
        this.featureMetaPath = new File(bundleRoot, "feature_meta.parquet").getAbsolutePath();
        this.statePath = new File(bundleRoot, "state.json").getAbsolutePath();
        this.bundleLogPath = new File(bundleRoot, "bundles.jsonl").getAbsolutePath();
        this.categoricalDictDir = new File(bundleRoot, "categorical_dictionaries").getAbsolutePath();

        this.featureKeyToId = new HashMap<String, Integer>();
        this.featureKeysInOrder = new ArrayList<String>();
        this.categoricalRegistries = new HashMap<String, CategoricalRegistry>();
        for (PointColumnSpec spec : this.pointSchema) {
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                this.categoricalRegistries.put(spec.name, new CategoricalRegistry());
            }
        }

        if (new File(this.statePath).exists()) {
            SessionState state = resumeStage(featureMetaPath, featureKeys);
            this.knownFeatureMode = state.knownFeatureMode;
            this.writesFeatureMeta = state.writesFeatureMeta;
            this.knownFeatureCount = state.knownFeatureCount;
            this.featureMetaSourcePath = state.featureMetaSourcePath;
            restoreCategoricalState(state.categoricalLabelsByColumn);
            this.lastCommittedSampleId = state.lastCommittedSampleId;
            this.committedBundleCount = state.committedBundleCount;
            this.cursorSampleId = state.nextExpectedSampleId;
            this.pendingBundleFirstSampleId = null;
            this.pendingBundleLastSampleId = null;
            this.pendingBundleSampleCount = 0;
            this.pendingBundleTraceCount = 0;
            this.openSampleId = null;
            this.currentSampleTraceCount = 0;
            this.bundlesFinalized = state.finishedStage;
            if (bundlesFinalized) {
                this.manifestPath = bundleManifestPath;
            }
            this.bundleWriter = new ArraySampleBundleWriter(
                    this.bundleOutDir,
                    this.bundleSampleMetaPath,
                    this.featureMetaPath,
                    this.nSamples,
                    this.bundleConfig,
                    this.pointSchema,
                    state.nextBundleId,
                    false);
        } else {
            SessionState state = initializeNewStage(bundleRoot, featureMetaPath, featureKeys);
            this.knownFeatureMode = state.knownFeatureMode;
            this.writesFeatureMeta = state.writesFeatureMeta;
            this.knownFeatureCount = state.knownFeatureCount;
            this.featureMetaSourcePath = state.featureMetaSourcePath;
            this.lastCommittedSampleId = null;
            this.committedBundleCount = 0;
            this.cursorSampleId = 0L;
            this.pendingBundleFirstSampleId = null;
            this.pendingBundleLastSampleId = null;
            this.pendingBundleSampleCount = 0;
            this.pendingBundleTraceCount = 0;
            this.openSampleId = null;
            this.currentSampleTraceCount = 0;
            this.bundleWriter = new ArraySampleBundleWriter(
                    this.bundleOutDir,
                    this.bundleSampleMetaPath,
                    this.featureMetaPath,
                    this.nSamples,
                    this.bundleConfig,
                    this.pointSchema,
                    0,
                    false);
            saveState();
        }
    }

    /**
     * 기본 설정으로 세션을 연다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema);
    }

    /**
     * build 옵션을 직접 지정해서 세션을 연다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema, buildOptions);
    }

    /**
     * known-feature metadata와 build 옵션을 같이 주고 세션을 연다.
     */
    public static ArrayDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            ArrayBinaryBuildOptions buildOptions) throws Exception {
        return new ArrayDatasetBuilder(outDir, sampleMetaPath, pointSchema, featureMetaPath, buildOptions);
    }

    /**
     * 현재 세션의 resume-safe 상태를 돌려준다.
     */
    public ArrayBuildSessionStatus status() {
        long nextExpected = resumeNextSampleId();
        return new ArrayBuildSessionStatus(
                lastCommittedSampleId,
                sampleKeyForId(lastCommittedSampleId),
                nextExpected,
                sampleKeyForId(Long.valueOf(nextExpected)),
                committedBundleCount,
                bundlesFinalized,
                bundlesFinalized ? bundleManifestPath : "",
                pendingBundleLastSampleId,
                sampleKeyForId(pendingBundleLastSampleId),
                openSampleId,
                sampleKeyForId(openSampleId));
    }

    /**
     * sample id 기준으로 trace를 묶어 쓰는 context를 연다.
     *
     * <p>array 입력은 보통 sample 하나 안에 trace 여러 개가 들어간다.
     * 이 context는 그 sample 경계를 public API에서 명시적으로 고정한다.
     * builder는 이 경계를 기준으로
     * - 다른 sample trace가 섞이는 실수를 막고
     * - sample이 완전히 닫힌 뒤에만 bundle checkpoint commit을 판단하고
     * - resume 시 {@code nextExpectedSampleId}와 ingestion 흐름을 일치시킨다.
     */
    public ArraySampleContext sample(long sampleId) throws Exception {
        ensureTraceStageOpen();
        return new ArraySampleContext(this, sampleId);
    }

    /**
     * sample key 기준으로 trace를 묶어 쓰는 context를 연다.
     *
     * <p>의미는 {@link #sample(long)}와 같고, 단지 외부 호출자가 sample key로
     * sample 경계를 지정할 수 있게 한 overload다.
     */
    public ArraySampleContext sample(String sampleKey) throws Exception {
        ensureTraceStageOpen();
        return new ArraySampleContext(this, resolveSampleId(null, sampleKey));
    }

    /**
     * trace 하나를 현재 sample에 추가한다.
     *
     * <p>top-level addTrace는 같은 sample 안에서만 연속 호출할 수 있다.
     * sample 경계를 넘기려면 {@link #sample(long)} 또는 {@link #sample(String)}
     * context를 써서 이전 sample을 명시적으로 닫아야 한다.
     *
     * <p>즉 array public API에서 sample context가 권장되는 이유는 문법 취향이 아니라,
     * sample 경계가 곧 checkpoint 경계이기 때문이다.
     */
    public void addTrace(long sampleId, Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        ensureTraceStageOpen();
        long resolvedSampleId = resolveSampleId(Long.valueOf(sampleId), null);
        if (openSampleId == null) {
            beginSample(resolvedSampleId);
        } else if (openSampleId.longValue() != resolvedSampleId) {
            throw new IllegalStateException(
                    "addTrace(...) crossed a sample boundary without closing the previous sample. "
                            + "Use sample(...) contexts or process traces for each sample together.");
        }
        int resolvedFeatureId = resolveFeatureId(featureId, featureKey);
        LinkedHashMap<String, Object> normalizedColumns = normalizeColumns(columns);
        bundleWriter.appendTrace(resolvedSampleId, resolvedFeatureId, normalizedColumns);
        currentSampleTraceCount += 1;
    }

    /**
     * finalized stage의 feature metadata에 새 컬럼을 merge한다.
     */
    public String updateFeatureMeta(List<Map<String, Object>> records, String on, boolean requireAll) throws Exception {
        finishStage();
        List<LinkedHashMap<String, Object>> baseRows = ArrayMetadataWriter.readRows(featureMetaPath);
        String joinCol = (on == null || on.isEmpty())
                ? (containsMetadataColumn(baseRows, this.buildOptions.featureKeyCol) ? this.buildOptions.featureKeyCol : "feature_id")
                : on;
        if (!containsMetadataColumn(baseRows, joinCol)) {
            throw new IllegalArgumentException("feature metadata join column not found: " + joinCol);
        }
        HashMap<Object, LinkedHashMap<String, Object>> updateByKey = new HashMap<Object, LinkedHashMap<String, Object>>();
        LinkedHashMap<String, String> newColumns = new LinkedHashMap<String, String>();
        for (Map<String, Object> record : records) {
            if (record == null) {
                throw new IllegalArgumentException("feature metadata update rows must not be null");
            }
            Object joinValue = record.get(joinCol);
            if (joinValue == null) {
                throw new IllegalArgumentException("update records must include join column: " + joinCol);
            }
            if (updateByKey.containsKey(joinValue)) {
                throw new IllegalArgumentException("duplicate feature metadata join key: " + joinValue);
            }
            LinkedHashMap<String, Object> copy = new LinkedHashMap<String, Object>();
            for (Map.Entry<String, Object> entry : record.entrySet()) {
                if (!entry.getKey().equals(joinCol) && containsMetadataColumn(baseRows, entry.getKey())) {
                    throw new IllegalArgumentException("feature metadata updates must add new columns only; overlapping column: " + entry.getKey());
                }
                copy.put(entry.getKey(), entry.getValue());
                if (!entry.getKey().equals(joinCol)) {
                    newColumns.put(entry.getKey(), entry.getKey());
                }
            }
            updateByKey.put(joinValue, copy);
        }

        ArrayList<Map<String, Object>> merged = new ArrayList<Map<String, Object>>(baseRows.size());
        for (LinkedHashMap<String, Object> baseRow : baseRows) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>(baseRow);
            LinkedHashMap<String, Object> update = updateByKey.get(baseRow.get(joinCol));
            for (String column : newColumns.keySet()) {
                Object value = (update == null) ? null : update.get(column);
                row.put(column, value);
            }
            if (requireAll) {
                for (String column : newColumns.keySet()) {
                    if (row.get(column) == null) {
                        throw new IllegalArgumentException("missing values remain in required feature metadata column: " + column);
                    }
                }
            }
            merged.add(row);
        }
        ArrayMetadataWriter.writeFeatureMeta(merged, featureMetaPath);
        return featureMetaPath;
    }

    /**
     * committed bundle들과 metadata를 바탕으로 stage manifest를 materialize한다.
     */
    public String finishStage() throws Exception {
        if (bundlesFinalized) {
            return bundleManifestPath;
        }
        ensureOpen();
        endSample(false);
        commitPendingBundle(true);
        writeFeatureMeta();
        writeCategoricalDictionaries();
        bundleWriter.updatePointSchema(pointSchema);
        bundleWriter.finish();
        bundlesFinalized = true;
        manifestPath = bundleManifestPath;
        saveState();
        return bundleManifestPath;
    }

    /**
     * legacy alias.
     */
    public String finishBundles() throws Exception {
        return finishStage();
    }

    /**
     * finalized stage를 바탕으로 최종 array binary shard를 만든다.
     */
    public String buildShards() throws Exception {
        return buildShards(false);
    }

    /**
     * finalized stage를 바탕으로 최종 array binary shard를 만든다.
     */
    public String buildShards(boolean cleanupBundles) throws Exception {
        return buildShards(cleanupBundles, false);
    }

    /**
     * build stats가 필요하면 같이 반환한다.
     */
    public String buildShards(boolean cleanupBundles, boolean returnStats) throws Exception {
        if (finished) {
            if (returnStats) {
                throw new IllegalStateException("array dataset builder has already built shards; build stats are no longer available");
            }
            return manifestPath;
        }
        ensureOpen();
        String stageManifestPath = finishStage();
        Object buildResult = ArrayShardBuilder.buildFromBundles(
                stageManifestPath,
                outDir,
                shardConfig,
                this.buildOptions.codec,
                this.buildOptions.sampleKeyCol,
                this.buildOptions.featureKeyCol);
        if (!(buildResult instanceof String)) {
            // current Java path always returns just the manifest path
            manifestPath = String.valueOf(buildResult);
        } else {
            manifestPath = (String) buildResult;
        }
        if (cleanupBundles) {
            deleteRecursively(new File(bundleOutDir));
        }
        finished = true;
        close();
        return manifestPath;
    }

    /**
     * legacy convenience alias.
     */
    public String finish() throws Exception {
        return buildShards(false);
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        try {
            if (!bundlesFinalized) {
                endSample(false);
                commitPendingBundle(true);
                saveState();
            }
        } finally {
            try {
                bundleWriter.close();
            } finally {
                closed = true;
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("array dataset builder is closed");
        }
    }

    private void ensureTraceStageOpen() {
        ensureOpen();
        if (bundlesFinalized) {
            throw new IllegalStateException("bundle stage has already been finalized");
        }
        if (finished) {
            throw new IllegalStateException("array shards have already been built");
        }
    }

    private long resolveSampleId(Long sampleId, String sampleKey) {
        if (sampleId == null && (sampleKey == null || sampleKey.isEmpty())) {
            throw new IllegalArgumentException("provide either sample_id or sample_key");
        }
        if (sampleId != null && sampleKey != null && !sampleKey.isEmpty()) {
            long resolved = resolveSampleId(null, sampleKey);
            if (sampleId.longValue() != resolved) {
                throw new IllegalArgumentException("sample_id/sample_key mismatch: " + sampleId + " != " + sampleKey);
            }
            return sampleId.longValue();
        }
        if (sampleId != null) {
            if (sampleId.longValue() < 0L || sampleId.longValue() >= nSamples) {
                throw new IllegalArgumentException("sample_id out of range: " + sampleId);
            }
            return sampleId.longValue();
        }
        Long resolved = sampleKeyToId.get(sampleKey);
        if (resolved == null) {
            throw new IllegalArgumentException("unknown sample key: " + sampleKey);
        }
        return resolved.longValue();
    }

    private void beginSample(long sampleId) {
        ensureTraceStageOpen();
        if (sampleId != cursorSampleId) {
            throw new IllegalArgumentException(
                    "array session expects sample_id " + cursorSampleId + "; got " + sampleId
                            + ". Resume from status().nextExpectedSampleId and process samples sequentially.");
        }
        if (openSampleId != null) {
            if (openSampleId.longValue() == sampleId) {
                return;
            }
            throw new IllegalStateException("another sample context is already open");
        }
        openSampleId = Long.valueOf(sampleId);
        currentSampleTraceCount = 0;
    }

    private void endSample(boolean abort) throws Exception {
        if (openSampleId == null) {
            return;
        }
        long sampleId = openSampleId.longValue();
        int traceCount = currentSampleTraceCount;
        openSampleId = null;
        currentSampleTraceCount = 0;
        if (abort) {
            return;
        }
        recordProcessedSample(sampleId, traceCount);
        commitPendingBundle(false);
        saveState();
    }

    private void recordProcessedSample(long sampleId, int traceCount) {
        if (pendingBundleFirstSampleId == null) {
            pendingBundleFirstSampleId = Long.valueOf(sampleId);
        }
        pendingBundleLastSampleId = Long.valueOf(sampleId);
        pendingBundleSampleCount += 1;
        pendingBundleTraceCount += traceCount;
        cursorSampleId = sampleId + 1L;
    }

    private void commitPendingBundle(boolean force) throws Exception {
        if (!force && !bundleWriter.shouldFlushBundle()) {
            return;
        }
        ArraySampleBundleWriter.BundleCommit commit = bundleWriter.flushBundle();
        if (commit == null) {
            return;
        }
        ObjectNode record = JsonUtils.objectNode();
        record.put("bundle_id", commit.bundleId);
        record.put("path", relativeTo(bundleOutDir, commit.path));
        record.put("first_sample_id", pendingBundleFirstSampleId.longValue());
        record.put("last_sample_id", pendingBundleLastSampleId.longValue());
        record.put("first_sample_key", sampleKeyForId(pendingBundleFirstSampleId));
        record.put("last_sample_key", sampleKeyForId(pendingBundleLastSampleId));
        record.put("sample_count", pendingBundleSampleCount);
        record.put("trace_count", pendingBundleTraceCount);
        record.put("row_count", commit.rowCount);
        record.put("byte_size", commit.byteSize);
        JsonUtils.appendJsonLine(bundleLogPath, record);

        lastCommittedSampleId = pendingBundleLastSampleId;
        committedBundleCount += 1;
        pendingBundleFirstSampleId = null;
        pendingBundleLastSampleId = null;
        pendingBundleSampleCount = 0;
        pendingBundleTraceCount = 0;
        saveState();
    }

    private long resumeNextSampleId() {
        if (lastCommittedSampleId == null) {
            return 0L;
        }
        return lastCommittedSampleId.longValue() + 1L;
    }

    private String sampleKeyForId(Long sampleId) {
        if (sampleId == null) {
            return "";
        }
        long idx = sampleId.longValue();
        if (idx < 0L || idx >= sampleKeys.size()) {
            return "";
        }
        String key = sampleKeys.get((int) idx);
        return (key == null) ? "" : key;
    }

    private void saveState() throws Exception {
        JsonUtils.writeJsonAtomic(statePath, statePayload());
    }

    private ObjectNode statePayload() {
        ObjectNode root = JsonUtils.objectNode();
        root.put("format_version", STATE_VERSION);
        root.put("stage_type", "array_bundle_stage_v1");
        root.put("sample_meta_path", sampleMetaPath);
        root.put("feature_meta_source_path", featureMetaSourcePath);
        root.set("build_options", buildOptionsPayload());
        ArrayNode schemaNode = root.putArray("point_schema");
        for (PointColumnSpec spec : pointSchema) {
            ObjectNode item = schemaNode.addObject();
            item.put("name", spec.name);
            item.put("storage_type", spec.storageType.value);
            item.put("logical_type", spec.logicalType.value);
        }
        root.put("known_feature_mode", knownFeatureMode);
        root.put("writes_feature_meta", writesFeatureMeta);
        if (knownFeatureCount == null) {
            root.putNull("known_feature_count");
        } else {
            root.put("known_feature_count", knownFeatureCount.intValue());
        }
        ArrayNode featureKeysNode = root.putArray("feature_keys_in_order");
        for (String featureKey : featureKeysInOrder) {
            featureKeysNode.add(featureKey);
        }
        ObjectNode categoricalNode = root.putObject("categorical_labels");
        for (Map.Entry<String, CategoricalRegistry> entry : categoricalRegistries.entrySet()) {
            ArrayNode labels = categoricalNode.putArray(entry.getKey());
            for (String label : entry.getValue().codeToLabel) {
                labels.add(label);
            }
        }
        root.put("next_bundle_id", bundleWriter.nextBundleId());
        if (lastCommittedSampleId == null) {
            root.putNull("last_committed_sample_id");
        } else {
            root.put("last_committed_sample_id", lastCommittedSampleId.longValue());
        }
        root.put("last_committed_sample_key", sampleKeyForId(lastCommittedSampleId));
        root.put("next_expected_sample_id", resumeNextSampleId());
        root.put("next_expected_sample_key", sampleKeyForId(Long.valueOf(resumeNextSampleId())));
        root.put("committed_bundle_count", committedBundleCount);
        root.put("finished_stage", bundlesFinalized);
        if (bundlesFinalized) {
            root.put("bundle_manifest_path", bundleManifestPath);
        } else {
            root.putNull("bundle_manifest_path");
        }
        if (pendingBundleLastSampleId == null) {
            root.putNull("buffered_through_sample_id");
            root.putNull("buffered_through_sample_key");
        } else {
            root.put("buffered_through_sample_id", pendingBundleLastSampleId.longValue());
            root.put("buffered_through_sample_key", sampleKeyForId(pendingBundleLastSampleId));
        }
        if (openSampleId == null) {
            root.putNull("in_progress_sample_id");
            root.putNull("in_progress_sample_key");
        } else {
            root.put("in_progress_sample_id", openSampleId.longValue());
            root.put("in_progress_sample_key", sampleKeyForId(openSampleId));
        }
        return root;
    }

    private ObjectNode buildOptionsPayload() {
        ObjectNode root = JsonUtils.objectNode();
        root.put("target_shard_mb", buildOptions.targetShardMb);
        if (buildOptions.nShards == null) {
            root.putNull("n_shards");
        } else {
            root.put("n_shards", buildOptions.nShards.intValue());
        }
        root.put("samples_per_block", buildOptions.samplesPerBlock);
        root.put("codec", buildOptions.codec);
        root.put("zstd_level", buildOptions.zstdLevel);
        root.put("sample_key_col", buildOptions.sampleKeyCol);
        root.put("feature_key_col", buildOptions.featureKeyCol);
        return root;
    }

    private SessionState initializeNewStage(File bundleRoot, String featureMetaPath, List<String> featureKeys) throws Exception {
        prepareEmptyDir(bundleRoot);
        Files.copy(new File(this.sampleMetaPath).toPath(), new File(this.bundleSampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);

        boolean localKnownFeatureMode;
        boolean localWritesFeatureMeta;
        Integer localKnownFeatureCount;
        String localFeatureMetaSourcePath = "";

        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            localKnownFeatureMode = true;
            localWritesFeatureMeta = false;
            File src = new File(featureMetaPath).getAbsoluteFile();
            localFeatureMetaSourcePath = src.getAbsolutePath();
            Files.copy(src.toPath(), new File(this.featureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(this.featureMetaPath);
            validateDenseIds(featureRows, "feature_id", "feature");
            localKnownFeatureCount = Integer.valueOf(featureRows.size());
            for (LinkedHashMap<String, Object> row : featureRows) {
                Object featureKey = row.get(this.buildOptions.featureKeyCol);
                if (featureKey != null) {
                    String key = featureKey.toString();
                    int id = ((Number) row.get("feature_id")).intValue();
                    featureKeyToId.put(key, Integer.valueOf(id));
                    featureKeysInOrder.add(key);
                }
            }
        } else if (featureKeys != null) {
            localKnownFeatureMode = true;
            localWritesFeatureMeta = true;
            int nextId = 0;
            for (String featureKey : featureKeys) {
                if (featureKeyToId.containsKey(featureKey)) {
                    throw new IllegalArgumentException("duplicate feature key: " + featureKey);
                }
                featureKeyToId.put(featureKey, Integer.valueOf(nextId++));
                featureKeysInOrder.add(featureKey);
            }
            localKnownFeatureCount = Integer.valueOf(featureKeysInOrder.size());
        } else {
            localKnownFeatureMode = false;
            localWritesFeatureMeta = true;
            localKnownFeatureCount = null;
        }
        return new SessionState(localKnownFeatureMode, localWritesFeatureMeta, localKnownFeatureCount, localFeatureMetaSourcePath, 0, null, 0, 0L, false, new HashMap<String, ArrayList<String>>());
    }

    private SessionState resumeStage(String featureMetaPath, List<String> featureKeys) throws Exception {
        JsonNode state = JsonUtils.readJson(statePath);
        validateResumeState(state, featureMetaPath, featureKeys);
        cleanupTmpFiles(new File(bundleOutDir));
        cleanupTmpFiles(new File(new File(bundleOutDir), "array_sample_bundles"));

        JsonNode featureKeysNode = state.get("feature_keys_in_order");
        if (featureKeysNode != null && featureKeysNode.isArray()) {
            for (JsonNode node : featureKeysNode) {
                String featureKey = node.asText();
                featureKeyToId.put(featureKey, Integer.valueOf(featureKeysInOrder.size()));
                featureKeysInOrder.add(featureKey);
            }
        }

        HashMap<String, ArrayList<String>> categoricalLabels = new HashMap<String, ArrayList<String>>();
        JsonNode categoricalNode = state.get("categorical_labels");
        if (categoricalNode != null && categoricalNode.isObject()) {
            java.util.Iterator<String> names = categoricalNode.fieldNames();
            while (names.hasNext()) {
                String name = names.next();
                JsonNode labelsNode = categoricalNode.get(name);
                ArrayList<String> labels = new ArrayList<String>();
                if (labelsNode != null && labelsNode.isArray()) {
                    for (JsonNode labelNode : labelsNode) {
                        labels.add(labelNode.asText());
                    }
                }
                categoricalLabels.put(name, labels);
            }
        }

        List<JsonNode> bundleRecords = JsonUtils.readJsonLines(bundleLogPath);
        Long localLastCommitted = bundleRecords.isEmpty()
                ? null
                : Long.valueOf(bundleRecords.get(bundleRecords.size() - 1).get("last_sample_id").asLong());
        long nextExpected = (localLastCommitted == null) ? 0L : localLastCommitted.longValue() + 1L;
        return new SessionState(
                state.path("known_feature_mode").asBoolean(false),
                state.path("writes_feature_meta").asBoolean(false),
                state.path("known_feature_count").isNumber() ? Integer.valueOf(state.path("known_feature_count").asInt()) : null,
                textOrEmpty(state, "feature_meta_source_path"),
                state.path("next_bundle_id").asInt(bundleRecords.size()),
                localLastCommitted,
                bundleRecords.size(),
                nextExpected,
                state.path("finished_stage").asBoolean(false),
                categoricalLabels);
    }

    private void validateResumeState(JsonNode state, String featureMetaPath, List<String> featureKeys) {
        if (!"array_bundle_stage_v1".equals(textOrEmpty(state, "stage_type"))) {
            throw new IllegalArgumentException("unsupported array build session type: " + textOrEmpty(state, "stage_type"));
        }
        if (!sampleMetaPath.equals(textOrEmpty(state, "sample_meta_path"))) {
            throw new IllegalArgumentException("sampleMetaPath does not match existing array build session");
        }
        if (!buildOptionsPayload().equals(state.get("build_options"))) {
            throw new IllegalArgumentException("buildOptions do not match existing array build session");
        }
        if (!pointSchemaEquals(state.get("point_schema"))) {
            throw new IllegalArgumentException("pointSchema does not match existing array build session");
        }
        String storedSource = textOrEmpty(state, "feature_meta_source_path");
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && !storedSource.isEmpty()) {
            String normalized = new File(featureMetaPath).getAbsolutePath();
            if (!normalized.equals(storedSource)) {
                throw new IllegalArgumentException("featureMetaPath does not match existing array build session");
            }
        }
        if (featureKeys != null) {
            JsonNode raw = state.get("feature_keys_in_order");
            if (raw == null || !raw.isArray() || raw.size() != featureKeys.size()) {
                throw new IllegalArgumentException("featureKeys do not match existing array build session");
            }
            for (int i = 0; i < featureKeys.size(); i++) {
                if (!featureKeys.get(i).equals(raw.get(i).asText())) {
                    throw new IllegalArgumentException("featureKeys do not match existing array build session");
                }
            }
        }
    }

    private boolean pointSchemaEquals(JsonNode raw) {
        if (raw == null || !raw.isArray() || raw.size() != pointSchema.size()) {
            return false;
        }
        for (int i = 0; i < pointSchema.size(); i++) {
            PointColumnSpec spec = pointSchema.get(i);
            JsonNode item = raw.get(i);
            if (!spec.name.equals(textOrEmpty(item, "name"))) {
                return false;
            }
            if (!spec.storageType.value.equals(textOrEmpty(item, "storage_type"))) {
                return false;
            }
            if (!spec.logicalType.value.equals(textOrEmpty(item, "logical_type"))) {
                return false;
            }
        }
        return true;
    }

    private void restoreCategoricalState(HashMap<String, ArrayList<String>> categoricalLabels) {
        for (Map.Entry<String, ArrayList<String>> entry : categoricalLabels.entrySet()) {
            CategoricalRegistry registry = categoricalRegistries.get(entry.getKey());
            if (registry == null) {
                continue;
            }
            registry.restore(entry.getValue());
        }
    }

    private int resolveFeatureId(Integer featureId, String featureKey) {
        if (featureId == null && (featureKey == null || featureKey.isEmpty())) {
            throw new IllegalArgumentException("provide either feature_id or feature_key");
        }
        if (featureId != null && featureKey != null && !featureKey.isEmpty()) {
            int resolved = resolveFeatureId(null, featureKey);
            if (featureId.intValue() != resolved) {
                throw new IllegalArgumentException("feature_id/feature_key mismatch: " + featureId + " != " + featureKey);
            }
            return featureId.intValue();
        }
        if (featureId != null) {
            if (!knownFeatureMode) {
                throw new IllegalArgumentException("discovered-feature mode requires feature_key inputs");
            }
            if (featureId.intValue() < 0) {
                throw new IllegalArgumentException("feature_id must be >= 0");
            }
            if (knownFeatureCount != null && featureId.intValue() >= knownFeatureCount.intValue()) {
                throw new IllegalArgumentException("feature_id out of range: " + featureId);
            }
            return featureId.intValue();
        }

        Integer resolved = featureKeyToId.get(featureKey);
        if (resolved != null) {
            return resolved.intValue();
        }
        if (knownFeatureMode) {
            throw new IllegalArgumentException("unknown feature key: " + featureKey);
        }
        int nextId = featureKeysInOrder.size();
        featureKeyToId.put(featureKey, Integer.valueOf(nextId));
        featureKeysInOrder.add(featureKey);
        return nextId;
    }

    private LinkedHashMap<String, Object> normalizeColumns(Map<String, Object> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("columns is required");
        }
        LinkedHashMap<String, Object> normalized = new LinkedHashMap<String, Object>();
        for (PointColumnSpec spec : pointSchema) {
            if (!columns.containsKey(spec.name)) {
                throw new IllegalArgumentException("missing point column: " + spec.name);
            }
            normalized.put(spec.name, normalizeColumnValues(spec, columns.get(spec.name)));
        }
        if (columns.size() != pointSchema.size()) {
            for (String columnName : columns.keySet()) {
                if (!containsColumn(pointSchema, columnName)) {
                    throw new IllegalArgumentException("unexpected point column: " + columnName);
                }
            }
        }
        int expectedLength = -1;
        for (Map.Entry<String, Object> entry : normalized.entrySet()) {
            int length = ArrayUtils.pointColumnLength(entry.getValue());
            if (expectedLength < 0) {
                expectedLength = length;
            } else if (expectedLength != length) {
                throw new IllegalArgumentException(
                        "point column length mismatch for " + entry.getKey() + ": expected=" + expectedLength + " got=" + length);
            }
        }
        return normalized;
    }

    private Object normalizeColumnValues(PointColumnSpec spec, Object values) {
        if (spec.logicalType == LogicalType.CATEGORICAL) {
            return encodeCategorical(spec.name, values);
        }
        if (spec.logicalType == LogicalType.TIMESTAMP_NS) {
            return normalizeTimestamp(values, spec.name);
        }
        if (spec.logicalType == LogicalType.TIMEDELTA_NS) {
            return normalizeTimedelta(values, spec.name);
        }
        switch (spec.storageType) {
            case FLOAT64:
                return ArrayUtils.toDoubleArray(values, spec.name);
            case INT64:
            case UINT64:
                return ArrayUtils.toLongArray(values, spec.name);
            case INT32:
            case UINT32:
                return ArrayUtils.toIntArray(values, spec.name);
            default:
                throw new IllegalArgumentException("unsupported storage type: " + spec.storageType);
        }
    }

    private Object encodeCategorical(String columnName, Object values) {
        CategoricalRegistry registry = categoricalRegistries.get(columnName);
        if (registry == null) {
            throw new IllegalStateException("missing categorical registry for column: " + columnName);
        }
        if (values == null) {
            return new long[0];
        }
        if (values instanceof String[]) {
            return registry.encode((String[]) values);
        }
        if (values instanceof List<?>) {
            return registry.encode((List<?>) values);
        }
        throw new IllegalArgumentException("categorical point column must be String[] or List<String>: " + columnName);
    }

    private long[] normalizeTimestamp(Object values, String columnName) {
        if (values == null) {
            return new long[0];
        }
        if (values instanceof Instant[]) {
            Instant[] source = (Instant[]) values;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i].getEpochSecond() * 1_000_000_000L + source[i].getNano();
            }
            return out;
        }
        if (values instanceof List<?>) {
            List<?> list = (List<?>) values;
            if (!list.isEmpty() && list.get(0) instanceof Instant) {
                long[] out = new long[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Instant instant = (Instant) list.get(i);
                    out[i] = instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
                }
                return out;
            }
        }
        return ArrayUtils.toLongArray(values, columnName);
    }

    private long[] normalizeTimedelta(Object values, String columnName) {
        if (values == null) {
            return new long[0];
        }
        if (values instanceof Duration[]) {
            Duration[] source = (Duration[]) values;
            long[] out = new long[source.length];
            for (int i = 0; i < source.length; i++) {
                out[i] = source[i].getSeconds() * 1_000_000_000L + source[i].getNano();
            }
            return out;
        }
        if (values instanceof List<?>) {
            List<?> list = (List<?>) values;
            if (!list.isEmpty() && list.get(0) instanceof Duration) {
                long[] out = new long[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Duration duration = (Duration) list.get(i);
                    out[i] = duration.getSeconds() * 1_000_000_000L + duration.getNano();
                }
                return out;
            }
        }
        return ArrayUtils.toLongArray(values, columnName);
    }

    private void writeFeatureMeta() throws Exception {
        if (!writesFeatureMeta) {
            return;
        }
        ArrayList<Map<String, Object>> records = new ArrayList<Map<String, Object>>(featureKeysInOrder.size());
        for (int featureId = 0; featureId < featureKeysInOrder.size(); featureId++) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("feature_id", featureId);
            row.put(this.buildOptions.featureKeyCol, featureKeysInOrder.get(featureId));
            records.add(row);
        }
        ArrayMetadataWriter.writeFeatureMeta(records, featureMetaPath);
    }

    private void writeCategoricalDictionaries() throws Exception {
        if (categoricalRegistries.isEmpty()) {
            return;
        }
        File dictRoot = new File(categoricalDictDir);
        if (!dictRoot.exists() && !dictRoot.mkdirs()) {
            throw new IllegalStateException("failed to create categorical dictionary dir: " + dictRoot.getAbsolutePath());
        }
        for (int i = 0; i < pointSchema.size(); i++) {
            PointColumnSpec spec = pointSchema.get(i);
            if (spec.logicalType != LogicalType.CATEGORICAL) {
                continue;
            }
            CategoricalRegistry registry = categoricalRegistries.get(spec.name);
            File dictPath = new File(dictRoot, spec.name + ".json");
            JsonUtils.writeCategoricalDictionary(dictPath.getAbsolutePath(), spec.name, registry.codeToLabel);
            pointSchema.set(i, spec.withDictionaryPath(dictPath.getAbsolutePath()));
        }
    }

    private static void validateDenseIds(List<LinkedHashMap<String, Object>> rows, String idCol, String entityName) {
        for (int i = 0; i < rows.size(); i++) {
            Object value = rows.get(i).get(idCol);
            if (!(value instanceof Number) || ((Number) value).longValue() != i) {
                throw new IllegalArgumentException(entityName + " metadata column '" + idCol + "' must equal dense row ids 0..N-1 in row order");
            }
        }
    }

    private static boolean containsColumn(List<PointColumnSpec> pointSchema, String name) {
        for (PointColumnSpec spec : pointSchema) {
            if (spec.name.equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static void validateBuildOptions(ArrayBinaryBuildOptions options) {
        if (options.samplesPerBlock <= 0) {
            throw new IllegalArgumentException("samplesPerBlock must be > 0");
        }
        if (options.targetShardMb <= 0 && (options.nShards == null || options.nShards.intValue() <= 0)) {
            throw new IllegalArgumentException("either targetShardMb or nShards must be > 0");
        }
        if (options.codec == null || !"none".equalsIgnoreCase(options.codec.trim())) {
            throw new IllegalArgumentException("java array builder currently supports only codec='none'");
        }
        if (options.sampleKeyCol == null || options.sampleKeyCol.isEmpty()) {
            throw new IllegalArgumentException("sampleKeyCol must not be empty");
        }
        if (options.featureKeyCol == null || options.featureKeyCol.isEmpty()) {
            throw new IllegalArgumentException("featureKeyCol must not be empty");
        }
    }

    private static ArrayShardConfig shardConfigFromOptions(ArrayBinaryBuildOptions options) {
        ArrayShardConfig cfg = new ArrayShardConfig();
        cfg.samplesPerBlock = options.samplesPerBlock;
        cfg.targetShardBytes = (long) options.targetShardMb * 1024L * 1024L;
        cfg.nShards = (options.nShards == null) ? 0 : options.nShards.intValue();
        return cfg;
    }

    private static boolean containsMetadataColumn(List<LinkedHashMap<String, Object>> rows, String name) {
        if (rows == null || rows.isEmpty()) {
            return false;
        }
        return rows.get(0).containsKey(name);
    }

    private static void prepareEmptyDir(File path) {
        if (path.exists()) {
            File[] children = path.listFiles();
            if (children != null && children.length > 0) {
                throw new IllegalArgumentException("bundleOutDir already exists and is not empty: " + path.getAbsolutePath());
            }
            return;
        }
        if (!path.mkdirs()) {
            throw new IllegalStateException("failed to create bundleOutDir: " + path.getAbsolutePath());
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

    private static void cleanupTmpFiles(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        File[] children = dir.listFiles();
        if (children == null) {
            return;
        }
        for (File child : children) {
            if (child.isFile() && child.getName().endsWith(".tmp")) {
                child.delete();
            }
        }
    }

    private static String relativeTo(String baseDir, String targetPath) {
        if (targetPath == null || targetPath.isEmpty()) {
            return "";
        }
        File target = new File(targetPath);
        if (!target.isAbsolute()) {
            return targetPath.replace("\\", "/");
        }
        return new File(baseDir).toPath().relativize(target.getAbsoluteFile().toPath()).toString().replace("\\", "/");
    }

    private static String textOrEmpty(JsonNode node, String fieldName) {
        JsonNode child = node.get(fieldName);
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static final class SessionState {
        final boolean knownFeatureMode;
        final boolean writesFeatureMeta;
        final Integer knownFeatureCount;
        final String featureMetaSourcePath;
        final int nextBundleId;
        final Long lastCommittedSampleId;
        final int committedBundleCount;
        final long nextExpectedSampleId;
        final boolean finishedStage;
        final HashMap<String, ArrayList<String>> categoricalLabelsByColumn;

        SessionState(
                boolean knownFeatureMode,
                boolean writesFeatureMeta,
                Integer knownFeatureCount,
                String featureMetaSourcePath,
                int nextBundleId,
                Long lastCommittedSampleId,
                int committedBundleCount,
                long nextExpectedSampleId,
                boolean finishedStage,
                HashMap<String, ArrayList<String>> categoricalLabelsByColumn) {
            this.knownFeatureMode = knownFeatureMode;
            this.writesFeatureMeta = writesFeatureMeta;
            this.knownFeatureCount = knownFeatureCount;
            this.featureMetaSourcePath = (featureMetaSourcePath == null) ? "" : featureMetaSourcePath;
            this.nextBundleId = nextBundleId;
            this.lastCommittedSampleId = lastCommittedSampleId;
            this.committedBundleCount = committedBundleCount;
            this.nextExpectedSampleId = nextExpectedSampleId;
            this.finishedStage = finishedStage;
            this.categoricalLabelsByColumn = categoricalLabelsByColumn;
        }
    }

    private static final class CategoricalRegistry {
        private final HashMap<String, Integer> labelToCode = new HashMap<String, Integer>();
        private final ArrayList<String> codeToLabel = new ArrayList<String>();

        long[] encode(String[] values) {
            long[] out = new long[(values == null) ? 0 : values.length];
            if (values == null) {
                return out;
            }
            for (int i = 0; i < values.length; i++) {
                String value = values[i];
                if (value == null) {
                    out[i] = 0L;
                    continue;
                }
                Integer code = labelToCode.get(value);
                if (code == null) {
                    code = Integer.valueOf(codeToLabel.size() + 1);
                    labelToCode.put(value, code);
                    codeToLabel.add(value);
                }
                out[i] = code.longValue();
            }
            return out;
        }

        long[] encode(List<?> values) {
            String[] labels = new String[(values == null) ? 0 : values.size()];
            if (values != null) {
                for (int i = 0; i < values.size(); i++) {
                    Object value = values.get(i);
                    labels[i] = (value == null) ? null : value.toString();
                }
            }
            return encode(labels);
        }

        void restore(List<String> labels) {
            labelToCode.clear();
            codeToLabel.clear();
            if (labels == null) {
                return;
            }
            for (int i = 0; i < labels.size(); i++) {
                String label = labels.get(i);
                labelToCode.put(label, Integer.valueOf(i + 1));
                codeToLabel.add(label);
            }
        }
    }

    /**
     * sample 하나에 속한 trace들을 묶어 쓰는 convenience context다.
     */
    public static final class ArraySampleContext implements AutoCloseable {
        private final ArrayDatasetBuilder builder;
        private final long sampleId;

        ArraySampleContext(ArrayDatasetBuilder builder, long sampleId) throws Exception {
            this.builder = builder;
            this.sampleId = sampleId;
            this.builder.beginSample(sampleId);
        }

        /**
         * 현재 sample에 trace 하나를 추가한다.
         *
         * <p>하나의 sample 안에 여러 feature trace가 들어갈 수 있으므로
         * context는 `addTrace(...)` 여러 번을 하나의 sample 단위로 묶는다.
         */
        public void addTrace(Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
            builder.addTrace(sampleId, featureId, featureKey, columns);
        }

        @Override
        public void close() throws Exception {
            builder.endSample(false);
        }
    }
}
