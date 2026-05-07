package fs.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import fs.config.BuildShardConfig;
import fs.io.common.ArrayMetadataWriter;
import fs.io.common.JsonUtils;
import fs.io.scalar.ScalarMetadataWriter;
import fs.io.scalar.ScalarSampleBundleManifestIO;
import fs.io.scalar.ScalarSampleBundleWriter;
import fs.io.scalar.ShardBuilder;
import fs.model.scalar.ScalarBuildSessionStatus;
import fs.model.scalar.ScalarSampleBundleManifest;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * scalar 값을 sample 단위로 받아 resumable sample-bundle stage를 만들고,
 * 그 stage를 최종 scalar shard로 변환하는 builder다.
 *
 * <p>자동 chunking은 내부 구현으로 숨겨지고, 사용자는 {@link #status()}로
 * 마지막 committed sample watermark를 확인한 뒤 {@link #writeSample(long, Map)}
 * 을 이어서 호출하면 된다.
 */
public class ScalarDatasetBuilder implements AutoCloseable {
    private static final int DEFAULT_BUNDLE_FLUSH_ROWS = 1_000_000;
    private static final int STATE_VERSION = 1;

    private final String outDir;
    private final String sourceSampleMetaPath;
    private final String sampleMajorOutDir;
    private final String sampleMajorBundlesDir;
    private final String sampleMajorManifestPath;
    private final String sampleMajorSampleMetaPath;
    private final String sampleMajorFeatureMetaPath;
    private final String statePath;
    private final String bundleLogPath;
    private final BuildShardConfig buildConfig;

    private final int nSamples;
    private final ArrayList<String> sampleKeys;
    private final LinkedHashMap<String, Integer> featureKeyToId;
    private final ArrayList<String> featureKeysInOrder;
    private final boolean knownFeatureMode;
    private final boolean writesFeatureMeta;
    private final Integer knownFeatureCount;
    private final String featureMetaSourcePath;
    private final ScalarSampleBundleWriter bundleWriter;

    private boolean closed;
    private boolean sampleMajorFinalized;
    private boolean shardsBuilt;
    private String manifestPath;

    private Long lastCommittedSampleId;
    private int committedBundleCount;
    private long cursorSampleId;
    private Long pendingBundleFirstSampleId;
    private Long pendingBundleLastSampleId;
    private int pendingBundleSampleCount;
    private int pendingBundleRowCount;

    /**
     * 기본 설정으로 새 세션을 열거나 기존 세션을 이어받는다.
     */
    public ScalarDatasetBuilder(String outDir, String sampleMetaPath) throws Exception {
        this(outDir, sampleMetaPath, "", null, null, "");
    }

    /**
     * 세부 옵션으로 새 세션을 열거나 기존 세션을 이어받는다.
     */
    public ScalarDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig,
            String sampleMajorOutDir) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        this.outDir = new File(outDir).getAbsolutePath();
        this.sourceSampleMetaPath = new File(sampleMetaPath).getAbsolutePath();
        this.buildConfig = (buildConfig == null) ? new BuildShardConfig() : buildConfig;
        this.closed = false;
        this.sampleMajorFinalized = false;
        this.shardsBuilt = false;
        this.manifestPath = "";

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sourceSampleMetaPath);
        validateDenseIds(sampleRows, this.buildConfig.sampleIdCol, "sample");
        for (String statsYCol : resolveStatsYCols(this.buildConfig)) {
            if (!containsColumn(sampleRows, statsYCol)) {
                throw new IllegalArgumentException("sample metadata is missing required target column: " + statsYCol);
            }
        }
        this.nSamples = sampleRows.size();
        this.sampleKeys = new ArrayList<String>(this.nSamples);
        for (LinkedHashMap<String, Object> row : sampleRows) {
            Object sampleKey = row.get(this.buildConfig.sampleKeyCol);
            this.sampleKeys.add(sampleKey == null ? null : sampleKey.toString());
        }

        File sampleMajorRoot = (sampleMajorOutDir == null || sampleMajorOutDir.isEmpty())
                ? new File(this.outDir, "sample_major_stage")
                : new File(sampleMajorOutDir);
        this.sampleMajorOutDir = sampleMajorRoot.getAbsolutePath();
        this.sampleMajorBundlesDir = new File(sampleMajorRoot, "sample_bundles").getAbsolutePath();
        this.sampleMajorManifestPath = new File(sampleMajorRoot, "sample_major_manifest.json").getAbsolutePath();
        this.sampleMajorSampleMetaPath = new File(sampleMajorRoot, "sample_meta.parquet").getAbsolutePath();
        this.sampleMajorFeatureMetaPath = new File(sampleMajorRoot, "feature_meta.parquet").getAbsolutePath();
        this.statePath = new File(sampleMajorRoot, "state.json").getAbsolutePath();
        this.bundleLogPath = new File(sampleMajorRoot, "bundles.jsonl").getAbsolutePath();

        this.featureKeyToId = new LinkedHashMap<String, Integer>();
        this.featureKeysInOrder = new ArrayList<String>();

        if (new File(this.statePath).exists()) {
            SessionState state = resumeStage(featureMetaPath, featureKeys);
            this.knownFeatureMode = state.knownFeatureMode;
            this.writesFeatureMeta = state.writesFeatureMeta;
            this.knownFeatureCount = state.knownFeatureCount;
            this.featureMetaSourcePath = state.featureMetaSourcePath;
            this.lastCommittedSampleId = state.lastCommittedSampleId;
            this.committedBundleCount = state.committedBundleCount;
            this.cursorSampleId = state.nextExpectedSampleId;
            this.pendingBundleFirstSampleId = null;
            this.pendingBundleLastSampleId = null;
            this.pendingBundleSampleCount = 0;
            this.pendingBundleRowCount = 0;
            this.sampleMajorFinalized = state.finishedStage;
            if (this.sampleMajorFinalized) {
                this.manifestPath = this.sampleMajorManifestPath;
            }
            this.bundleWriter = new ScalarSampleBundleWriter(this.sampleMajorBundlesDir, DEFAULT_BUNDLE_FLUSH_ROWS, state.nextBundleId, false);
        } else {
            SessionState state = initializeNewStage(sampleMajorRoot, featureMetaPath, featureKeys);
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
            this.pendingBundleRowCount = 0;
            this.bundleWriter = new ScalarSampleBundleWriter(this.sampleMajorBundlesDir, DEFAULT_BUNDLE_FLUSH_ROWS, 0, false);
            saveState();
        }
    }

    /**
     * 기본 설정으로 세션을 연다.
     */
    public static ScalarDatasetBuilder openSession(String outDir, String sampleMetaPath) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath);
    }

    /**
     * 세부 옵션으로 세션을 연다.
     */
    public static ScalarDatasetBuilder openSession(
            String outDir,
            String sampleMetaPath,
            String featureMetaPath,
            List<String> featureKeys,
            BuildShardConfig buildConfig,
            String sampleMajorOutDir) throws Exception {
        return new ScalarDatasetBuilder(outDir, sampleMetaPath, featureMetaPath, featureKeys, buildConfig, sampleMajorOutDir);
    }

    /**
     * 현재 세션의 resume-safe 상태를 돌려준다.
     */
    public ScalarBuildSessionStatus status() {
        long nextExpected = resumeNextSampleId();
        return new ScalarBuildSessionStatus(
                lastCommittedSampleId,
                sampleKeyForId(lastCommittedSampleId),
                nextExpected,
                sampleKeyForId(Long.valueOf(nextExpected)),
                committedBundleCount,
                sampleMajorFinalized,
                sampleMajorFinalized ? sampleMajorManifestPath : "",
                pendingBundleLastSampleId,
                sampleKeyForId(pendingBundleLastSampleId));
    }

    /**
     * sample 하나를 완결된 단위로 기록한다.
     *
     * <p>public ingest 단위는 sample 하나다. per-value write path는 열어두지 않는다.
     */
    public void writeSample(long sampleId, Map<?, ?> values) throws Exception {
        ensureSampleMajorOpen();
        if (sampleId != cursorSampleId) {
            throw new IllegalArgumentException(
                    "scalar session expects sample_id " + cursorSampleId + "; got " + sampleId
                            + ". Resume from status().nextExpectedSampleId and write samples sequentially.");
        }
        LinkedHashMap<Integer, Double> featureValues = new LinkedHashMap<Integer, Double>();
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            Double normalized = normalizeScalarValue(entry.getValue());
            int featureId = resolveFeatureId(entry.getKey());
            if (featureValues.containsKey(Integer.valueOf(featureId))) {
                throw new IllegalArgumentException("duplicate feature assignment within sample " + sampleId + ": feature_id=" + featureId);
            }
            if (normalized != null) {
                featureValues.put(Integer.valueOf(featureId), normalized);
            }
        }
        bundleWriter.appendSample(sampleId, featureValues);
        recordProcessedSample(sampleId, featureValues.size());
        commitPendingBundle(false);
    }

    /**
     * scalar session에서는 sample context를 공개하지 않는다.
     */
    public Object openSample(long sampleId) {
        throw new IllegalStateException("scalar sample contexts are disabled; use writeSample(sampleId, values)");
    }

    /**
     * discovered-feature mode에서 자동 생성된 feature metadata에 새 컬럼을 merge한다.
     */
    public String updateFeatureMeta(List<Map<String, Object>> records, String on, boolean requireAll) throws Exception {
        finishStage();
        List<LinkedHashMap<String, Object>> baseRows = ArrayMetadataWriter.readRows(sampleMajorFeatureMetaPath);
        String joinCol = (on == null || on.isEmpty())
                ? (containsColumn(baseRows, buildConfig.featureKeyCol) ? buildConfig.featureKeyCol : buildConfig.featureIdCol)
                : on;
        if (!containsColumn(baseRows, joinCol)) {
            throw new IllegalArgumentException("feature metadata join column not found: " + joinCol);
        }

        LinkedHashMap<Object, LinkedHashMap<String, Object>> updatesByKey = new LinkedHashMap<Object, LinkedHashMap<String, Object>>();
        LinkedHashMap<String, String> newColumns = new LinkedHashMap<String, String>();
        for (Map<String, Object> record : records) {
            if (record == null) {
                throw new IllegalArgumentException("feature metadata update rows must not be null");
            }
            Object joinValue = record.get(joinCol);
            if (joinValue == null) {
                throw new IllegalArgumentException("update records must include join column: " + joinCol);
            }
            if (updatesByKey.containsKey(joinValue)) {
                throw new IllegalArgumentException("duplicate feature metadata join key: " + joinValue);
            }
            LinkedHashMap<String, Object> copy = new LinkedHashMap<String, Object>();
            for (Map.Entry<String, Object> entry : record.entrySet()) {
                if (!entry.getKey().equals(joinCol) && containsColumn(baseRows, entry.getKey())) {
                    throw new IllegalArgumentException("feature metadata updates must add new columns only; overlapping column: " + entry.getKey());
                }
                copy.put(entry.getKey(), entry.getValue());
                if (!entry.getKey().equals(joinCol)) {
                    newColumns.put(entry.getKey(), entry.getKey());
                }
            }
            updatesByKey.put(joinValue, copy);
        }

        ArrayList<Map<String, Object>> merged = new ArrayList<Map<String, Object>>(baseRows.size());
        for (LinkedHashMap<String, Object> baseRow : baseRows) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>(baseRow);
            LinkedHashMap<String, Object> update = updatesByKey.get(baseRow.get(joinCol));
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
        ScalarMetadataWriter.writeFeatureMeta(merged, sampleMajorFeatureMetaPath);
        return sampleMajorFeatureMetaPath;
    }

    /**
     * committed bundle들과 metadata를 바탕으로 sample-major stage manifest를 materialize한다.
     */
    public String finishStage() throws Exception {
        if (sampleMajorFinalized) {
            return sampleMajorManifestPath;
        }
        ensureOpen();
        commitPendingBundle(true);
        writeFeatureMeta();
        copySampleMeta();
        bundleWriter.finish();
        List<String> bundlePaths = committedBundlePaths();
        ScalarSampleBundleManifestIO.write(
                new ScalarSampleBundleManifest(
                        sampleMajorSampleMetaPath,
                        sampleMajorFeatureMetaPath,
                        bundlePaths,
                        buildConfig.sampleIdCol,
                        buildConfig.featureIdCol,
                        buildConfig.valueCol),
                sampleMajorManifestPath);
        sampleMajorFinalized = true;
        manifestPath = sampleMajorManifestPath;
        saveState();
        return sampleMajorManifestPath;
    }

    /**
     * legacy alias.
     */
    public String finishSampleMajor() throws Exception {
        return finishStage();
    }

    /**
     * sample-major stage를 바탕으로 최종 scalar shard dataset을 만든다.
     */
    public String buildShards() throws Exception {
        return buildShards(false);
    }

    /**
     * sample-major stage를 바탕으로 최종 scalar shard dataset을 만든다.
     */
    public String buildShards(boolean keepSampleMajor) throws Exception {
        if (shardsBuilt) {
            return manifestPath;
        }
        ensureOpen();
        String stageManifestPath = finishStage();
        manifestPath = ShardBuilder.buildShardsFromSampleBundles(stageManifestPath, outDir, buildConfig);
        if (!keepSampleMajor) {
            deleteRecursively(new File(sampleMajorOutDir));
        }
        shardsBuilt = true;
        close();
        return manifestPath;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        try {
            if (!sampleMajorFinalized) {
                commitPendingBundle(true);
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
            throw new IllegalStateException("scalar dataset builder is closed");
        }
    }

    private void ensureSampleMajorOpen() {
        ensureOpen();
        if (sampleMajorFinalized) {
            throw new IllegalStateException("sample-major stage has already been finalized");
        }
        if (shardsBuilt) {
            throw new IllegalStateException("scalar shards have already been built");
        }
    }

    private void recordProcessedSample(long sampleId, int rowCount) {
        if (pendingBundleFirstSampleId == null) {
            pendingBundleFirstSampleId = Long.valueOf(sampleId);
        }
        pendingBundleLastSampleId = Long.valueOf(sampleId);
        pendingBundleSampleCount += 1;
        pendingBundleRowCount += rowCount;
        cursorSampleId = sampleId + 1L;
    }

    private void commitPendingBundle(boolean force) throws Exception {
        if (!force && !bundleWriter.shouldFlushBundle()) {
            return;
        }
        ScalarSampleBundleWriter.BundleCommit commit = bundleWriter.flushBundle();
        if (commit == null) {
            return;
        }
        ObjectNode record = JsonUtils.objectNode();
        record.put("bundle_id", commit.bundleId);
        record.put("path", relativeTo(sampleMajorOutDir, commit.path));
        record.put("first_sample_id", pendingBundleFirstSampleId.longValue());
        record.put("last_sample_id", pendingBundleLastSampleId.longValue());
        record.put("first_sample_key", sampleKeyForId(pendingBundleFirstSampleId));
        record.put("last_sample_key", sampleKeyForId(pendingBundleLastSampleId));
        record.put("sample_count", pendingBundleSampleCount);
        record.put("row_count", pendingBundleRowCount);
        JsonUtils.appendJsonLine(bundleLogPath, record);

        lastCommittedSampleId = pendingBundleLastSampleId;
        committedBundleCount += 1;
        pendingBundleFirstSampleId = null;
        pendingBundleLastSampleId = null;
        pendingBundleSampleCount = 0;
        pendingBundleRowCount = 0;
        saveState();
    }

    private Long resumeNextCommittedSampleId() {
        return (lastCommittedSampleId == null) ? Long.valueOf(0L) : Long.valueOf(lastCommittedSampleId.longValue() + 1L);
    }

    private long resumeNextSampleId() {
        return resumeNextCommittedSampleId().longValue();
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
        root.put("stage_type", "scalar_bundle_stage_v1");
        root.put("sample_meta_path", sourceSampleMetaPath);
        root.put("feature_meta_source_path", featureMetaSourcePath);
        root.set("build_config", buildConfigPayload());
        root.put("known_feature_mode", knownFeatureMode);
        root.put("writes_feature_meta", writesFeatureMeta);
        if (knownFeatureCount == null) {
            root.putNull("known_feature_count");
        } else {
            root.put("known_feature_count", knownFeatureCount.intValue());
        }
        if (shouldPersistFeatureKeysInState()) {
            ArrayNode featureKeysNode = root.putArray("feature_keys_in_order");
            for (String featureKey : featureKeysInOrder) {
                featureKeysNode.add(featureKey);
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
        root.put("next_expected_sample_key", sampleKeyForId(resumeNextCommittedSampleId()));
        root.put("committed_bundle_count", committedBundleCount);
        root.put("finished_stage", sampleMajorFinalized);
        if (sampleMajorFinalized) {
            root.put("bundle_manifest_path", sampleMajorManifestPath);
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
        return root;
    }

    /**
     * state.json에 feature key 전체 목록을 유지해야 하는지 결정한다.
     *
     * <p>discovered-feature mode나 `featureKeys` 리스트 기반 known-feature mode는
     * session state만으로 feature 순서를 복구해야 하므로 key 목록을 저장한다.
     *
     * <p>반면 외부 `feature_meta_path`를 받은 known-feature mode는 authoritative feature metadata가
     * 이미 따로 있으므로, 매 sample마다 큰 key 배열을 state에 다시 쓰지 않아도 된다.
     */
    private boolean shouldPersistFeatureKeysInState() {
        return !knownFeatureMode || featureMetaSourcePath.isEmpty();
    }

    private ObjectNode buildConfigPayload() {
        ObjectNode node = JsonUtils.objectNode();
        node.put("n_shards", buildConfig.nShards);
        node.put("target_shard_bytes", buildConfig.targetShardBytes);
        node.put("feature_id_col", buildConfig.featureIdCol);
        node.put("value_col", buildConfig.valueCol);
        node.put("sample_id_col", buildConfig.sampleIdCol);
        node.put("sample_key_col", buildConfig.sampleKeyCol);
        node.put("feature_key_col", buildConfig.featureKeyCol);
        node.put("feature_meta_path", buildConfig.featureMetaPath);
        node.put("path_col", buildConfig.pathCol);
        node.put("y_col", buildConfig.yCol);
        ArrayNode statsYCols = node.putArray("stats_y_cols");
        for (String value : resolveStatsYCols(buildConfig)) {
            statsYCols.add(value);
        }
        node.put("values_type", buildConfig.valuesType);
        node.put("valid_type", buildConfig.validType);
        return node;
    }

    private SessionState initializeNewStage(File sampleMajorRoot, String featureMetaPath, List<String> featureKeys) throws Exception {
        prepareEmptyDir(sampleMajorRoot);
        File bundlesDir = new File(sampleMajorBundlesDir);
        if (!bundlesDir.exists() && !bundlesDir.mkdirs()) {
            throw new IllegalStateException("failed to create sample-bundle dir: " + bundlesDir.getAbsolutePath());
        }
        copySampleMeta();

        boolean localKnownFeatureMode;
        boolean localWritesFeatureMeta;
        Integer localKnownFeatureCount;
        String localFeatureMetaSourcePath = "";

        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            localKnownFeatureMode = true;
            localWritesFeatureMeta = false;
            File src = new File(featureMetaPath).getAbsoluteFile();
            localFeatureMetaSourcePath = src.getAbsolutePath();
            Files.copy(src.toPath(), new File(sampleMajorFeatureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(sampleMajorFeatureMetaPath);
            validateDenseIds(featureRows, buildConfig.featureIdCol, "feature");
            localKnownFeatureCount = Integer.valueOf(featureRows.size());
            if (containsColumn(featureRows, buildConfig.featureKeyCol)) {
                for (LinkedHashMap<String, Object> row : featureRows) {
                    String key = row.get(buildConfig.featureKeyCol).toString();
                    int id = ((Number) row.get(buildConfig.featureIdCol)).intValue();
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
        return new SessionState(localKnownFeatureMode, localWritesFeatureMeta, localKnownFeatureCount, localFeatureMetaSourcePath, 0, null, 0, 0L, false);
    }

    private SessionState resumeStage(String featureMetaPath, List<String> featureKeys) throws Exception {
        JsonNode state = JsonUtils.readJson(statePath);
        validateResumeState(state, featureMetaPath, featureKeys);
        cleanupTmpFiles(new File(sampleMajorOutDir));
        cleanupTmpFiles(new File(sampleMajorBundlesDir));

        JsonNode storedFeatureKeys = state.get("feature_keys_in_order");
        if (storedFeatureKeys != null && storedFeatureKeys.isArray()) {
            for (JsonNode node : storedFeatureKeys) {
                String featureKey = node.asText();
                featureKeyToId.put(featureKey, Integer.valueOf(featureKeysInOrder.size()));
                featureKeysInOrder.add(featureKey);
            }
        } else if (state.path("known_feature_mode").asBoolean(false)
                && !textOrEmpty(state, "feature_meta_source_path").isEmpty()) {
            loadFeatureKeysFromKnownMeta();
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
                state.path("finished_stage").asBoolean(false));
    }

    private void validateResumeState(JsonNode state, String featureMetaPath, List<String> featureKeys) {
        if (!"scalar_bundle_stage_v1".equals(textOrEmpty(state, "stage_type"))) {
            throw new IllegalArgumentException("unsupported scalar build session type: " + textOrEmpty(state, "stage_type"));
        }
        if (!sourceSampleMetaPath.equals(textOrEmpty(state, "sample_meta_path"))) {
            throw new IllegalArgumentException("sampleMetaPath does not match existing scalar build session");
        }
        if (!buildConfigMatches(state.get("build_config"))) {
            throw new IllegalArgumentException("buildConfig does not match existing scalar build session");
        }
        String storedSource = textOrEmpty(state, "feature_meta_source_path");
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && !storedSource.isEmpty()) {
            String normalized = new File(featureMetaPath).getAbsolutePath();
            if (!normalized.equals(storedSource)) {
                throw new IllegalArgumentException("featureMetaPath does not match existing scalar build session");
            }
        }
        if (featureKeys != null) {
            JsonNode raw = state.get("feature_keys_in_order");
            if (raw == null || !raw.isArray() || raw.size() != featureKeys.size()) {
                throw new IllegalArgumentException("featureKeys do not match existing scalar build session");
            }
            for (int i = 0; i < featureKeys.size(); i++) {
                if (!featureKeys.get(i).equals(raw.get(i).asText())) {
                    throw new IllegalArgumentException("featureKeys do not match existing scalar build session");
                }
            }
        }
    }

    /**
     * known-feature mode에서 state에 key 목록을 저장하지 않았을 때, copied feature metadata에서
     * `feature_key -> feature_id` 순서를 다시 복구한다.
     */
    private void loadFeatureKeysFromKnownMeta() throws Exception {
        List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(sampleMajorFeatureMetaPath);
        validateDenseIds(featureRows, buildConfig.featureIdCol, "feature");
        for (LinkedHashMap<String, Object> row : featureRows) {
            Object keyValue = row.get(buildConfig.featureKeyCol);
            if (keyValue == null) {
                continue;
            }
            int featureId = ((Number) row.get(buildConfig.featureIdCol)).intValue();
            String featureKey = keyValue.toString();
            while (featureKeysInOrder.size() <= featureId) {
                featureKeysInOrder.add(null);
            }
            featureKeysInOrder.set(featureId, featureKey);
            featureKeyToId.put(featureKey, Integer.valueOf(featureId));
        }
        for (int featureId = 0; featureId < featureKeysInOrder.size(); featureId++) {
            if (featureKeysInOrder.get(featureId) == null) {
                throw new IllegalArgumentException(
                        "feature metadata is missing feature key for dense feature_id=" + featureId);
            }
        }
    }

    private boolean buildConfigMatches(JsonNode raw) {
        if (raw == null || !raw.isObject()) {
            return false;
        }
        if (raw.path("n_shards").asInt(0) != buildConfig.nShards) {
            return false;
        }
        if (raw.path("target_shard_bytes").asLong(0L) != buildConfig.targetShardBytes) {
            return false;
        }
        if (!buildConfig.featureIdCol.equals(textOrEmpty(raw, "feature_id_col"))) {
            return false;
        }
        if (!buildConfig.valueCol.equals(textOrEmpty(raw, "value_col"))) {
            return false;
        }
        if (!buildConfig.sampleIdCol.equals(textOrEmpty(raw, "sample_id_col"))) {
            return false;
        }
        if (!buildConfig.sampleKeyCol.equals(textOrEmpty(raw, "sample_key_col"))) {
            return false;
        }
        if (!buildConfig.featureKeyCol.equals(textOrEmpty(raw, "feature_key_col"))) {
            return false;
        }
        if (!textOrEmpty(raw, "feature_meta_path").equals(nullToEmpty(buildConfig.featureMetaPath))) {
            return false;
        }
        if (!buildConfig.pathCol.equals(textOrEmpty(raw, "path_col"))) {
            return false;
        }
        if (!buildConfig.yCol.equals(textOrEmpty(raw, "y_col"))) {
            return false;
        }
        if (!buildConfig.valuesType.equals(textOrEmpty(raw, "values_type"))) {
            return false;
        }
        if (!buildConfig.validType.equals(textOrEmpty(raw, "valid_type"))) {
            return false;
        }
        JsonNode statsNode = raw.get("stats_y_cols");
        List<String> expected = resolveStatsYCols(buildConfig);
        if (statsNode == null || !statsNode.isArray() || statsNode.size() != expected.size()) {
            return false;
        }
        for (int i = 0; i < expected.size(); i++) {
            if (!expected.get(i).equals(statsNode.get(i).asText())) {
                return false;
            }
        }
        return true;
    }

    private List<String> committedBundlePaths() throws Exception {
        ArrayList<String> out = new ArrayList<String>();
        for (JsonNode node : JsonUtils.readJsonLines(bundleLogPath)) {
            out.add(resolveAgainst(sampleMajorOutDir, textOrEmpty(node, "path")));
        }
        return out;
    }

    private Double normalizeScalarValue(Object rawValue) {
        if (rawValue == null) {
            return null;
        }
        double value = ((Number) rawValue).doubleValue();
        if (Double.isNaN(value)) {
            return null;
        }
        return Double.valueOf(value);
    }

    private int resolveFeatureId(Object featureRef) {
        if (featureRef instanceof Number) {
            int featureId = ((Number) featureRef).intValue();
            if (!knownFeatureMode) {
                throw new IllegalArgumentException("feature_id cannot be used in discovered-feature mode; use feature_key instead");
            }
            if (featureId < 0) {
                throw new IllegalArgumentException("feature_id out of range: " + featureId);
            }
            if (knownFeatureCount != null && featureId >= knownFeatureCount.intValue()) {
                throw new IllegalArgumentException("feature_id out of range: " + featureId);
            }
            return featureId;
        }
        String featureKey = String.valueOf(featureRef);
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

    private void writeFeatureMeta() throws Exception {
        if (!writesFeatureMeta) {
            return;
        }
        ArrayList<Map<String, Object>> records = new ArrayList<Map<String, Object>>(featureKeysInOrder.size());
        for (int featureId = 0; featureId < featureKeysInOrder.size(); featureId++) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            row.put(buildConfig.featureIdCol, featureId);
            row.put(buildConfig.featureKeyCol, featureKeysInOrder.get(featureId));
            records.add(row);
        }
        ScalarMetadataWriter.writeFeatureMeta(records, sampleMajorFeatureMetaPath);
    }

    private void copySampleMeta() throws Exception {
        if (new File(sourceSampleMetaPath).getCanonicalPath().equals(new File(sampleMajorSampleMetaPath).getCanonicalPath())) {
            return;
        }
        Files.copy(new File(sourceSampleMetaPath).toPath(), new File(sampleMajorSampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
    }

    private static List<String> resolveStatsYCols(BuildShardConfig config) {
        ArrayList<String> out = new ArrayList<String>();
        if (config.statsYCols != null) {
            for (String value : config.statsYCols) {
                if (value != null && !value.isEmpty() && !out.contains(value)) {
                    out.add(value);
                }
            }
        }
        if (out.isEmpty()) {
            out.add(config.yCol);
        }
        return out;
    }

    private static void validateDenseIds(List<LinkedHashMap<String, Object>> rows, String idCol, String entityName) {
        for (int i = 0; i < rows.size(); i++) {
            Object value = rows.get(i).get(idCol);
            if (!(value instanceof Number) || ((Number) value).longValue() != i) {
                throw new IllegalArgumentException(entityName + " metadata column '" + idCol + "' must equal dense row ids 0..N-1 in row order");
            }
        }
    }

    private static boolean containsColumn(List<LinkedHashMap<String, Object>> rows, String name) {
        if (rows == null || rows.isEmpty()) {
            return false;
        }
        return rows.get(0).containsKey(name);
    }

    private static void prepareEmptyDir(File path) {
        if (path.exists()) {
            File[] children = path.listFiles();
            if (children != null && children.length > 0) {
                throw new IllegalArgumentException("sampleMajorOutDir already exists and is not empty: " + path.getAbsolutePath());
            }
            return;
        }
        if (!path.mkdirs()) {
            throw new IllegalStateException("failed to create sampleMajorOutDir: " + path.getAbsolutePath());
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
        return (child == null || child.isNull()) ? "" : child.asText();
    }

    private static String nullToEmpty(String value) {
        return (value == null) ? "" : value;
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

        SessionState(
                boolean knownFeatureMode,
                boolean writesFeatureMeta,
                Integer knownFeatureCount,
                String featureMetaSourcePath,
                int nextBundleId,
                Long lastCommittedSampleId,
                int committedBundleCount,
                long nextExpectedSampleId,
                boolean finishedStage) {
            this.knownFeatureMode = knownFeatureMode;
            this.writesFeatureMeta = writesFeatureMeta;
            this.knownFeatureCount = knownFeatureCount;
            this.featureMetaSourcePath = (featureMetaSourcePath == null) ? "" : featureMetaSourcePath;
            this.nextBundleId = nextBundleId;
            this.lastCommittedSampleId = lastCommittedSampleId;
            this.committedBundleCount = committedBundleCount;
            this.nextExpectedSampleId = nextExpectedSampleId;
            this.finishedStage = finishedStage;
        }
    }
}
