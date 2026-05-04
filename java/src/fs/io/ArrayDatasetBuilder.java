package fs.io;

import fs.config.ArrayBundleConfig;
import fs.config.ArrayShardConfig;
import fs.model.LogicalType;
import fs.model.PointColumnSpec;
import fs.model.StorageType;

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

public class ArrayDatasetBuilder implements AutoCloseable {
    private final String outDir;
    private final String sampleMetaPath;
    private final String bundleOutDir;
    private final String bundleSampleMetaPath;
    private final String featureMetaPath;
    private final ArrayShardConfig shardConfig;
    private final ArrayBundleConfig bundleConfig;
    private final List<PointColumnSpec> pointSchema;
    private final boolean knownFeatureMode;
    private final boolean writesFeatureMeta;
    private final HashMap<String, Integer> featureKeyToId;
    private final ArrayList<String> featureKeysInOrder;
    private final Integer knownFeatureCount;
    private final HashMap<String, CategoricalRegistry> categoricalRegistries;
    private final ArraySampleBundleWriter bundleWriter;
    private final int nSamples;

    private boolean closed;
    private boolean bundlesFinalized;
    private boolean finished;
    private String manifestPath;
    private String bundleManifestPath;

    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema) throws Exception {
        this(outDir, sampleMetaPath, pointSchema, "", null, null, null, "");
    }

    public ArrayDatasetBuilder(
            String outDir,
            String sampleMetaPath,
            List<PointColumnSpec> pointSchema,
            String featureMetaPath,
            List<String> featureKeys,
            ArrayShardConfig shardConfig,
            ArrayBundleConfig bundleConfig,
            String bundleOutDir) throws Exception {
        if (featureMetaPath != null && !featureMetaPath.isEmpty() && featureKeys != null) {
            throw new IllegalArgumentException("provide at most one of featureMetaPath or featureKeys");
        }
        this.outDir = new File(outDir).getAbsolutePath();
        this.sampleMetaPath = new File(sampleMetaPath).getAbsolutePath();
        this.pointSchema = PointColumnSpec.normalizeList(pointSchema);
        this.shardConfig = (shardConfig == null) ? new ArrayShardConfig() : shardConfig;
        this.bundleConfig = (bundleConfig == null) ? new ArrayBundleConfig() : bundleConfig;
        this.closed = false;
        this.bundlesFinalized = false;
        this.finished = false;
        this.manifestPath = "";
        this.bundleManifestPath = "";

        List<LinkedHashMap<String, Object>> sampleRows = ArrayMetadataWriter.readRows(this.sampleMetaPath);
        validateDenseIds(sampleRows, "sample_id", "sample");
        this.nSamples = sampleRows.size();

        File bundleRoot = (bundleOutDir == null || bundleOutDir.isEmpty())
                ? new File(this.outDir, "bundle_stage")
                : new File(bundleOutDir);
        this.bundleOutDir = bundleRoot.getAbsolutePath();
        prepareEmptyDir(bundleRoot);

        this.bundleSampleMetaPath = new File(bundleRoot, "sample_meta.parquet").getAbsolutePath();
        Files.copy(new File(this.sampleMetaPath).toPath(), new File(this.bundleSampleMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);

        this.featureMetaPath = new File(bundleRoot, "feature_meta.parquet").getAbsolutePath();
        this.featureKeyToId = new HashMap<String, Integer>();
        this.featureKeysInOrder = new ArrayList<String>();
        this.categoricalRegistries = new HashMap<String, CategoricalRegistry>();
        for (PointColumnSpec spec : this.pointSchema) {
            if (spec.logicalType == LogicalType.CATEGORICAL) {
                this.categoricalRegistries.put(spec.name, new CategoricalRegistry());
            }
        }

        if (featureMetaPath != null && !featureMetaPath.isEmpty()) {
            this.knownFeatureMode = true;
            this.writesFeatureMeta = false;
            File src = new File(featureMetaPath).getAbsoluteFile();
            Files.copy(src.toPath(), new File(this.featureMetaPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            List<LinkedHashMap<String, Object>> featureRows = ArrayMetadataWriter.readRows(this.featureMetaPath);
            validateDenseIds(featureRows, "feature_id", "feature");
            this.knownFeatureCount = Integer.valueOf(featureRows.size());
            for (LinkedHashMap<String, Object> row : featureRows) {
                Object featureKey = row.get(ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL);
                if (featureKey != null) {
                    String key = featureKey.toString();
                    int id = ((Number) row.get("feature_id")).intValue();
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

        this.bundleWriter = new ArraySampleBundleWriter(
                this.bundleOutDir,
                this.bundleSampleMetaPath,
                this.featureMetaPath,
                this.nSamples,
                this.bundleConfig,
                this.pointSchema);
    }

    public ArraySampleContext sample(long sampleId) {
        ensureTraceStageOpen();
        return new ArraySampleContext(this, sampleId);
    }

    public void addTrace(long sampleId, Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
        ensureTraceStageOpen();
        if (sampleId < 0L || sampleId >= nSamples) {
            throw new IllegalArgumentException("sample_id out of range: " + sampleId);
        }
        int resolvedFeatureId = resolveFeatureId(featureId, featureKey);
        LinkedHashMap<String, Object> normalizedColumns = normalizeColumns(columns);
        bundleWriter.appendTrace(sampleId, resolvedFeatureId, normalizedColumns);
    }

    public String finishBundles() throws Exception {
        if (bundlesFinalized) {
            return bundleManifestPath;
        }
        ensureOpen();
        writeFeatureMeta();
        writeCategoricalDictionaries();
        bundleWriter.updatePointSchema(pointSchema);
        bundleManifestPath = bundleWriter.finish();
        bundlesFinalized = true;
        return bundleManifestPath;
    }

    public String updateFeatureMeta(List<Map<String, Object>> records, String on, boolean requireAll) throws Exception {
        finishBundles();
        List<LinkedHashMap<String, Object>> baseRows = ArrayMetadataWriter.readRows(featureMetaPath);
        String joinCol = (on == null || on.isEmpty())
                ? (containsMetadataColumn(baseRows, ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL) ? ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL : "feature_id")
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

    public String buildShards() throws Exception {
        return buildShards(false);
    }

    public String buildShards(boolean cleanupBundles) throws Exception {
        if (finished) {
            return manifestPath;
        }
        ensureOpen();
        String bundleManifest = finishBundles();
        manifestPath = ArrayShardBuilder.buildFromBundles(bundleManifest, outDir, shardConfig);
        if (cleanupBundles) {
            deleteRecursively(new File(bundleOutDir));
        }
        finished = true;
        closed = true;
        return manifestPath;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            return;
        }
        try {
            if (!bundlesFinalized) {
                try {
                    bundleWriter.close();
                } catch (Exception ignored) {
                    // best-effort cleanup path
                }
                deleteRecursively(new File(bundleOutDir));
            }
        } finally {
            closed = true;
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
        if (spec.storageType == StorageType.FLOAT64) {
            return ArrayUtils.toDoubleArray(values, spec.name);
        }
        if (spec.storageType == StorageType.INT32) {
            return ArrayUtils.toIntArray(values, spec.name);
        }
        return ArrayUtils.toLongArray(values, spec.name);
    }

    private long[] encodeCategorical(String columnName, Object values) {
        CategoricalRegistry registry = categoricalRegistries.get(columnName);
        if (values == null) {
            return new long[0];
        }
        if (values instanceof long[] || values instanceof int[] || values instanceof Number[] || values instanceof List<?>) {
            if (values instanceof List<?>) {
                List<?> list = (List<?>) values;
                boolean allNumbers = true;
                for (Object item : list) {
                    if (item != null && !(item instanceof Number)) {
                        allNumbers = false;
                        break;
                    }
                }
                if (allNumbers) {
                    return ArrayUtils.toLongArray(values, columnName);
                }
                return registry.encode(list);
            }
            if (values instanceof Number[]) {
                return ArrayUtils.toLongArray(values, columnName);
            }
            return ArrayUtils.toLongArray(values, columnName);
        }
        if (values instanceof String[]) {
            return registry.encode((String[]) values);
        }
        throw new IllegalArgumentException("unsupported categorical values for column " + columnName + ": " + values.getClass().getName());
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
            row.put(ArrayBinaryFormat.DEFAULT_FEATURE_KEY_COL, featureKeysInOrder.get(featureId));
            records.add(row);
        }
        ArrayMetadataWriter.writeFeatureMeta(records, featureMetaPath);
    }

    private void writeCategoricalDictionaries() throws Exception {
        if (categoricalRegistries.isEmpty()) {
            return;
        }
        File dictRoot = new File(bundleOutDir, "categorical_dictionaries");
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

    }

    public static final class ArraySampleContext implements AutoCloseable {
        private final ArrayDatasetBuilder builder;
        private final long sampleId;

        ArraySampleContext(ArrayDatasetBuilder builder, long sampleId) {
            this.builder = builder;
            this.sampleId = sampleId;
        }

        public void addTrace(Integer featureId, String featureKey, Map<String, Object> columns) throws Exception {
            builder.addTrace(sampleId, featureId, featureKey, columns);
        }

        @Override
        public void close() {
            // sample-scoped helper does not own resources
        }
    }
}
