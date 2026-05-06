package fs.io;

import fs.config.BuildShardConfig;
import fs.model.scalar.ScalarSampleBundleManifest;
import fs.io.common.ArrayMetadataWriter;
import fs.io.scalar.ScalarSampleBundleManifestIO;
import fs.io.scalar.ScalarMetadataWriter;
import fs.io.scalar.ScalarSampleBundleWriter;
import fs.io.scalar.ShardBuilder;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * scalar 값을 직접 받아 sample-major stage와 최종 scalar shard dataset을 만드는 builder다.
 *
 * <p>public API의 기본 단위는 sample이다. 사용자는
 * {@link #writeSample(long, Map)} 로 sample 하나를 한 번에 쓰거나,
 * {@link #openSample(long)} 으로 sample-scoped context를 열어 값을 순차적으로 추가할 수 있다.
 */
public class ScalarDatasetBuilder implements AutoCloseable {
    private static final int DEFAULT_BUNDLE_FLUSH_ROWS = 1_000_000;

    private final String outDir;
    private final String sourceSampleMetaPath;
    private final String sampleMajorOutDir;
    private final String sampleMajorBundlesDir;
    private final String sampleMajorManifestPath;
    private final String sampleMajorSampleMetaPath;
    private final String sampleMajorFeatureMetaPath;
    private final BuildShardConfig buildConfig;

    private final int nSamples;
    private final boolean[] sampleWritten;
    private final LinkedHashMap<String, Integer> featureKeyToId;
    private final ArrayList<String> featureKeysInOrder;
    private final boolean knownFeatureMode;
    private final boolean writesFeatureMeta;
    private final Integer knownFeatureCount;
    private final ScalarSampleBundleWriter bundleWriter;

    private boolean closed;
    private boolean sampleMajorFinalized;
    private boolean shardsBuilt;
    private String manifestPath;
    private long openSampleId;
    private LinkedHashMap<Integer, Double> openSampleValues;

    /**
     * 기본 설정의 builder를 생성한다.
     */
    public ScalarDatasetBuilder(String outDir, String sampleMetaPath) throws Exception {
        this(outDir, sampleMetaPath, "", null, null, null);
    }

    /**
     * builder를 전체 옵션과 함께 생성한다.
     *
     * @param outDir 최종 shard 출력 디렉터리
     * @param sampleMetaPath dense sample metadata parquet 경로
     * @param featureMetaPath known-feature mode에서 사용할 feature metadata parquet 경로
     * @param featureKeys known-feature mode에서 사용할 feature key 목록
     * @param buildConfig sample-major stage를 shard로 변환할 때 사용할 build 설정
     * @param sampleMajorOutDir intermediate sample-major stage 디렉터리
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
        this.openSampleId = -1L;
        this.openSampleValues = null;

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sourceSampleMetaPath);
        validateDenseIds(sampleRows, this.buildConfig.sampleIdCol, "sample");
        if (!containsColumn(sampleRows, this.buildConfig.yCol)) {
            throw new IllegalArgumentException("sample metadata is missing required target column: " + this.buildConfig.yCol);
        }
        for (String statsYCol : resolveStatsYCols(this.buildConfig)) {
            if (!containsColumn(sampleRows, statsYCol)) {
                throw new IllegalArgumentException("sample metadata is missing required target column: " + statsYCol);
            }
        }
        this.nSamples = sampleRows.size();
        this.sampleWritten = new boolean[this.nSamples];

        File sampleMajorRoot = (sampleMajorOutDir == null || sampleMajorOutDir.isEmpty())
                ? new File(this.outDir, "sample_major_stage")
                : new File(sampleMajorOutDir);
        prepareEmptyDir(sampleMajorRoot);
        this.sampleMajorOutDir = sampleMajorRoot.getAbsolutePath();
        this.sampleMajorBundlesDir = new File(sampleMajorRoot, "sample_bundles").getAbsolutePath();
        File bundlesDir = new File(this.sampleMajorBundlesDir);
        if (!bundlesDir.exists() && !bundlesDir.mkdirs()) {
            throw new IllegalStateException("failed to create sample-bundle dir: " + bundlesDir.getAbsolutePath());
        }

        this.sampleMajorManifestPath = new File(sampleMajorRoot, "sample_major_manifest.json").getAbsolutePath();
        this.sampleMajorSampleMetaPath = new File(sampleMajorRoot, "sample_meta.parquet").getAbsolutePath();
        this.sampleMajorFeatureMetaPath = new File(sampleMajorRoot, "feature_meta.parquet").getAbsolutePath();

        this.featureKeyToId = new LinkedHashMap<String, Integer>();
        this.featureKeysInOrder = new ArrayList<String>();
        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            this.knownFeatureMode = true;
            this.writesFeatureMeta = false;
            File src = new File(featureMetaPath).getAbsoluteFile();
            Files.copy(src.toPath(), new File(this.sampleMajorFeatureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(this.sampleMajorFeatureMetaPath);
            validateDenseIds(featureRows, this.buildConfig.featureIdCol, "feature");
            this.knownFeatureCount = Integer.valueOf(featureRows.size());
            if (containsColumn(featureRows, this.buildConfig.featureKeyCol)) {
                for (LinkedHashMap<String, Object> row : featureRows) {
                    String key = row.get(this.buildConfig.featureKeyCol).toString();
                    int id = ((Number) row.get(this.buildConfig.featureIdCol)).intValue();
                    featureKeyToId.put(key, Integer.valueOf(id));
                    featureKeysInOrder.add(key);
                }
            }
        } else if (featureKeys != null) {
            this.knownFeatureMode = true;
            this.writesFeatureMeta = true;
            int nextId = 0;
            for (String featureKey : featureKeys) {
                if (featureKeyToId.containsKey(featureKey)) {
                    throw new IllegalArgumentException("duplicate feature key: " + featureKey);
                }
                featureKeyToId.put(featureKey, Integer.valueOf(nextId++));
                featureKeysInOrder.add(featureKey);
            }
            this.knownFeatureCount = Integer.valueOf(featureKeysInOrder.size());
        } else {
            this.knownFeatureMode = false;
            this.writesFeatureMeta = true;
            this.knownFeatureCount = null;
        }

        this.bundleWriter = new ScalarSampleBundleWriter(this.sampleMajorBundlesDir, DEFAULT_BUNDLE_FLUSH_ROWS);
    }

    /**
     * sample 하나를 완결된 단위로 기록한다.
     *
     * @param sampleId dense sample id
     * @param values feature별 scalar value 매핑
     */
    public void writeSample(long sampleId, Map<?, ?> values) throws Exception {
        beginSample(sampleId);
        try {
            writeValues(values);
        } catch (Exception e) {
            endSample(true);
            throw e;
        }
        endSample(false);
    }

    /**
     * sample 하나에 값을 순차적으로 쓰는 context를 연다.
     */
    public ScalarSampleContext openSample(long sampleId) {
        return new ScalarSampleContext(this, sampleId);
    }

    /**
     * 자동 생성된 feature metadata에 새 컬럼을 merge한다.
     */
    public String updateFeatureMeta(List<Map<String, Object>> records, String on, boolean requireAll) throws Exception {
        finishSampleMajor();
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
     * sample-major stage를 finalize하고 stage manifest를 작성한다.
     *
     * @return sample-major manifest 경로
     */
    public String finishSampleMajor() throws Exception {
        if (sampleMajorFinalized) {
            return sampleMajorManifestPath;
        }
        ensureOpen();
        if (openSampleValues != null) {
            endSample(false);
        }
        writeFeatureMeta();
        copySampleMeta();
        List<String> bundlePaths = bundleWriter.finish();
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
        return sampleMajorManifestPath;
    }

    /**
     * sample-major stage를 바탕으로 최종 scalar shard dataset을 만든다.
     */
    public String buildShards() throws Exception {
        return buildShards(false);
    }

    /**
     * sample-major stage를 바탕으로 최종 scalar shard dataset을 만든다.
     *
     * @param keepSampleMajor true면 intermediate sample-major stage를 남긴다
     * @return 최종 shard manifest 경로
     */
    public String buildShards(boolean keepSampleMajor) throws Exception {
        if (shardsBuilt) {
            return manifestPath;
        }
        ensureOpen();
        String stageManifestPath = finishSampleMajor();
        manifestPath = ShardBuilder.buildShardsFromSampleBundles(stageManifestPath, outDir, buildConfig);
        if (!keepSampleMajor) {
            deleteRecursively(new File(sampleMajorOutDir));
        }
        shardsBuilt = true;
        closed = true;
        return manifestPath;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        try {
            if (!sampleMajorFinalized) {
                try {
                    bundleWriter.close();
                } catch (Exception ignored) {
                }
                deleteRecursively(new File(sampleMajorOutDir));
            }
        } finally {
            closed = true;
        }
    }

    /**
     * sample 쓰기 세션을 연다.
     *
     * <p>sample 재방문을 막기 위해 이미 기록된 sample이면 즉시 실패한다.
     */
    private void beginSample(long sampleId) {
        ensureSampleMajorOpen();
        if (sampleId < 0L || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
        if (sampleWritten[(int) sampleId]) {
            throw new IllegalArgumentException("sample_id has already been written and cannot be revisited: " + sampleId);
        }
        if (openSampleValues != null) {
            throw new IllegalStateException("another sample context is already open");
        }
        openSampleId = sampleId;
        openSampleValues = new LinkedHashMap<Integer, Double>();
    }

    /**
     * 현재 열린 sample에 feature/value 하나를 추가한다.
     */
    private void writeValue(Object featureRef, Object rawValue) {
        if (openSampleValues == null) {
            throw new IllegalStateException("no sample context is currently open");
        }
        Double value = normalizeScalarValue(rawValue);
        int featureId = resolveFeatureId(featureRef);
        if (openSampleValues.containsKey(Integer.valueOf(featureId))) {
            throw new IllegalArgumentException("duplicate feature assignment within sample " + openSampleId + ": feature_id=" + featureId);
        }
        if (value != null) {
            openSampleValues.put(Integer.valueOf(featureId), value);
        }
    }

    private void writeValues(Map<?, ?> values) {
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            writeValue(entry.getKey(), entry.getValue());
        }
    }

    /**
     * 현재 sample 쓰기 세션을 닫고 bundle writer로 넘긴다.
     */
    private void endSample(boolean abort) throws Exception {
        if (openSampleValues == null) {
            return;
        }
        long sampleId = openSampleId;
        LinkedHashMap<Integer, Double> values = openSampleValues;
        openSampleValues = null;
        openSampleId = -1L;
        if (abort) {
            return;
        }
        bundleWriter.appendSample(sampleId, values);
        sampleWritten[(int) sampleId] = true;
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

    /**
     * feature id 또는 key 입력을 dense feature id로 정규화한다.
     */
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

    /**
     * discovered-feature mode에서 feature metadata를 자동 생성한다.
     */
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

    /**
     * source sample metadata를 sample-major stage 위치로 복사한다.
     */
    private void copySampleMeta() throws Exception {
        if (new File(sourceSampleMetaPath).getCanonicalPath().equals(new File(sampleMajorSampleMetaPath).getCanonicalPath())) {
            return;
        }
        Files.copy(new File(sourceSampleMetaPath).toPath(), new File(sampleMajorSampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
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

    public static final class ScalarSampleContext implements AutoCloseable {
        private final ScalarDatasetBuilder builder;
        private final long sampleId;

        ScalarSampleContext(ScalarDatasetBuilder builder, long sampleId) {
            this.builder = builder;
            this.sampleId = sampleId;
            this.builder.beginSample(sampleId);
        }

        public void writeValue(Object feature, Object value) {
            builder.writeValue(feature, value);
        }

        public void writeValues(Map<?, ?> values) {
            builder.writeValues(values);
        }

        @Override
        public void close() throws Exception {
            builder.endSample(false);
        }
    }
}
