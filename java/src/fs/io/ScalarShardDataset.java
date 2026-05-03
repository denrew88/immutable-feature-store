package fs.io;

import fs.model.Feature;
import fs.model.FeatureLocation;
import fs.model.RowBatch;
import fs.model.ScalarFeatureValues;
import fs.model.ScalarValue;
import fs.model.ShardManifest;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.NoSuchElementException;

public final class ScalarShardDataset implements AutoCloseable {
    private static final int DEFAULT_BATCH_SIZE = 128;

    private final ShardManifest manifest;
    private final FeatureLocatorIndex locator;
    private final SampleIdIndex sampleIds;
    private final FeatureIdIndex featureIds;
    private final DuckDBShardReader reader;

    public ScalarShardDataset(String manifestPath) throws Exception {
        this(ManifestIO.read(manifestPath));
    }

    public ScalarShardDataset(ShardManifest manifest) throws Exception {
        this.manifest = manifest;
        this.locator = FeatureLocatorIndex.load(manifest);
        this.sampleIds = SampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
        this.featureIds = FeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
        this.reader = new DuckDBShardReader(manifest);
    }

    public ShardManifest manifest() {
        return manifest;
    }

    public ScalarFeatureValues getValues(int featureId, long[] sampleIds) throws SQLException {
        FeatureRequest request = normalizeFeatureRequests(new int[]{featureId}, null, true).get(0);
        ResolvedSamples resolvedSamples = resolveSampleIds(sampleIds);
        Feature feature = loadFeature(request.featureId);
        return buildFeatureValues(request, resolvedSamples, feature);
    }

    public ScalarFeatureValues getValuesBySampleKeys(int featureId, String[] sampleKeys) throws SQLException {
        FeatureRequest request = normalizeFeatureRequests(new int[]{featureId}, null, true).get(0);
        ResolvedSamples resolvedSamples = resolveSampleKeys(sampleKeys);
        Feature feature = loadFeature(request.featureId);
        return buildFeatureValues(request, resolvedSamples, feature);
    }

    public ScalarFeatureValues getValuesByKeys(String featureKey, String[] sampleKeys) throws SQLException {
        ArrayList<FeatureRequest> requests = normalizeFeatureKeyRequests(new String[]{featureKey}, true);
        ResolvedSamples resolvedSamples = resolveSampleKeys(sampleKeys);
        Feature feature = loadFeature(requests.get(0).featureId);
        return buildFeatureValues(requests.get(0), resolvedSamples, feature);
    }

    public ScalarValue getValue(int featureId, long sampleId) throws SQLException {
        return getValues(featureId, new long[]{sampleId}).values.get(0);
    }

    public ScalarValue getValueByKey(String featureKey, String sampleKey) throws SQLException {
        return getValuesByKeys(featureKey, new String[]{sampleKey}).values.get(0);
    }

    public Iterable<ScalarFeatureValues> iterMany(int[] requestedFeatureIds, long[] requestedSampleIds) throws SQLException {
        return iterMany(requestedFeatureIds, requestedSampleIds, DEFAULT_BATCH_SIZE, true);
    }

    public Iterable<ScalarFeatureValues> iterMany(
            int[] requestedFeatureIds,
            long[] requestedSampleIds,
            int batchSize,
            boolean maintainOrder) throws SQLException {
        final ArrayList<FeatureRequest> requests = normalizeFeatureRequests(requestedFeatureIds, null, maintainOrder);
        final ResolvedSamples resolvedSamples = resolveSampleIds(requestedSampleIds);
        final int resolvedBatchSize = validateBatchSize(batchSize);
        return new Iterable<ScalarFeatureValues>() {
            @Override
            public Iterator<ScalarFeatureValues> iterator() {
                return new FeatureValuesIterator(requests, resolvedSamples, resolvedBatchSize);
            }
        };
    }

    public Iterable<ScalarFeatureValues> iterManyByKey(String[] requestedFeatureKeys, String[] requestedSampleKeys) throws SQLException {
        return iterManyByKey(requestedFeatureKeys, requestedSampleKeys, DEFAULT_BATCH_SIZE, true);
    }

    public Iterable<ScalarFeatureValues> iterManyByKey(
            String[] requestedFeatureKeys,
            String[] requestedSampleKeys,
            int batchSize,
            boolean maintainOrder) throws SQLException {
        final ArrayList<FeatureRequest> requests = normalizeFeatureKeyRequests(requestedFeatureKeys, maintainOrder);
        final ResolvedSamples resolvedSamples = resolveSampleKeys(requestedSampleKeys);
        final int resolvedBatchSize = validateBatchSize(batchSize);
        return new Iterable<ScalarFeatureValues>() {
            @Override
            public Iterator<ScalarFeatureValues> iterator() {
                return new FeatureValuesIterator(requests, resolvedSamples, resolvedBatchSize);
            }
        };
    }

    @Override
    public void close() throws Exception {
        SQLException closeEx = null;
        try {
            locator.close();
        } catch (SQLException e) {
            closeEx = e;
        }
        try {
            reader.close();
        } catch (SQLException e) {
            if (closeEx == null) {
                closeEx = e;
            }
        }
        if (closeEx != null) {
            throw closeEx;
        }
    }

    private Feature loadFeature(int featureId) throws SQLException {
        FeatureLocation location = locator.find(featureId);
        if (location == null) {
            throw new IllegalArgumentException("feature_id is not present in shard locator: " + featureId);
        }
        return reader.loadFeatureByOffset(location.shardId, location.offsetInShard);
    }

    private ScalarFeatureValues buildFeatureValues(FeatureRequest request, ResolvedSamples resolvedSamples, Feature feature) {
        ArrayList<ScalarValue> out = new ArrayList<ScalarValue>(resolvedSamples.sampleIds.length);
        for (int i = 0; i < resolvedSamples.sampleIds.length; i++) {
            long sampleId = resolvedSamples.sampleIds[i];
            String sampleKey = resolvedSamples.sampleKeys[i];
            int idx = (int) sampleId;
            boolean present = idx < feature.valid.length && feature.valid[idx] != 0;
            Double value = present ? Double.valueOf(feature.values[idx]) : null;
            out.add(new ScalarValue(sampleId, sampleKey, present, value));
        }
        return new ScalarFeatureValues(request.featureId, request.featureKey, out);
    }

    private ArrayList<FeatureRequest> normalizeFeatureRequests(int[] requestedFeatureIds, String[] featureKeysOverride, boolean maintainOrder) {
        ArrayList<FeatureRequest> requests = new ArrayList<FeatureRequest>(requestedFeatureIds.length);
        for (int i = 0; i < requestedFeatureIds.length; i++) {
            Integer resolvedFeatureId = featureIds.findFeatureId(requestedFeatureIds[i]);
            if (resolvedFeatureId == null) {
                throw new IllegalArgumentException("unknown feature_id: " + requestedFeatureIds[i]);
            }
            FeatureLocation location = locator.find(resolvedFeatureId.intValue());
            if (location == null) {
                throw new IllegalArgumentException("feature_id is not present in shard locator: " + resolvedFeatureId.intValue());
            }
            String featureKey = featureKeysOverride != null ? featureKeysOverride[i] : featureIds.keyForId(resolvedFeatureId.intValue());
            requests.add(new FeatureRequest(
                    resolvedFeatureId.intValue(),
                    featureKey,
                    location.shardId,
                    location.offsetInShard
            ));
        }
        if (!maintainOrder) {
            Collections.sort(requests, FEATURE_REQUEST_COMPARATOR);
        }
        return requests;
    }

    private ArrayList<FeatureRequest> normalizeFeatureKeyRequests(String[] requestedFeatureKeys, boolean maintainOrder) {
        int[] featureIdsArray = new int[requestedFeatureKeys.length];
        String[] featureKeysOverride = new String[requestedFeatureKeys.length];
        for (int i = 0; i < requestedFeatureKeys.length; i++) {
            Integer featureId = featureIds.findFeatureIdByKey(requestedFeatureKeys[i]);
            if (featureId == null) {
                throw new IllegalArgumentException("unknown feature_key: " + requestedFeatureKeys[i]);
            }
            featureIdsArray[i] = featureId.intValue();
            featureKeysOverride[i] = requestedFeatureKeys[i];
        }
        return normalizeFeatureRequests(featureIdsArray, featureKeysOverride, maintainOrder);
    }

    private ResolvedSamples resolveSampleIds(long[] requestedSampleIds) {
        long[] resolvedIds = new long[requestedSampleIds.length];
        String[] resolvedKeys = new String[requestedSampleIds.length];
        for (int i = 0; i < requestedSampleIds.length; i++) {
            Long resolvedSampleId = sampleIds.findSampleId(requestedSampleIds[i]);
            if (resolvedSampleId == null) {
                throw new IllegalArgumentException("unknown sample_id: " + requestedSampleIds[i]);
            }
            resolvedIds[i] = resolvedSampleId.longValue();
            resolvedKeys[i] = sampleIds.keyForId(resolvedIds[i]);
        }
        return new ResolvedSamples(resolvedIds, resolvedKeys);
    }

    private ResolvedSamples resolveSampleKeys(String[] requestedSampleKeys) {
        long[] resolvedIds = new long[requestedSampleKeys.length];
        String[] resolvedKeys = new String[requestedSampleKeys.length];
        for (int i = 0; i < requestedSampleKeys.length; i++) {
            Long resolvedSampleId = sampleIds.findSampleIdByKey(requestedSampleKeys[i]);
            if (resolvedSampleId == null) {
                throw new IllegalArgumentException("unknown sample_key: " + requestedSampleKeys[i]);
            }
            resolvedIds[i] = resolvedSampleId.longValue();
            resolvedKeys[i] = requestedSampleKeys[i];
        }
        return new ResolvedSamples(resolvedIds, resolvedKeys);
    }

    private static int validateBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        return batchSize;
    }

    private final class FeatureValuesIterator implements Iterator<ScalarFeatureValues> {
        private final List<FeatureRequest> requests;
        private final ResolvedSamples resolvedSamples;
        private final int batchSize;
        private final ArrayList<ScalarFeatureValues> currentBatch;
        private int requestIndex;
        private int batchIndex;

        FeatureValuesIterator(List<FeatureRequest> requests, ResolvedSamples resolvedSamples, int batchSize) {
            this.requests = requests;
            this.resolvedSamples = resolvedSamples;
            this.batchSize = batchSize;
            this.currentBatch = new ArrayList<ScalarFeatureValues>(batchSize);
            this.requestIndex = 0;
            this.batchIndex = 0;
        }

        @Override
        public boolean hasNext() {
            ensureBatchLoaded();
            return batchIndex < currentBatch.size();
        }

        @Override
        public ScalarFeatureValues next() {
            ensureBatchLoaded();
            if (batchIndex >= currentBatch.size()) {
                throw new NoSuchElementException();
            }
            ScalarFeatureValues value = currentBatch.get(batchIndex);
            batchIndex += 1;
            return value;
        }

        private void ensureBatchLoaded() {
            if (batchIndex < currentBatch.size()) {
                return;
            }
            if (requestIndex >= requests.size()) {
                return;
            }
            loadNextBatch();
        }

        private void loadNextBatch() {
            currentBatch.clear();
            batchIndex = 0;
            int end = Math.min(requestIndex + batchSize, requests.size());
            List<FeatureRequest> chunk = requests.subList(requestIndex, end);
            LinkedHashMap<Integer, ArrayList<FeatureRequest>> shardGroups = new LinkedHashMap<Integer, ArrayList<FeatureRequest>>();
            for (FeatureRequest request : chunk) {
                ArrayList<FeatureRequest> group = shardGroups.get(Integer.valueOf(request.shardId));
                if (group == null) {
                    group = new ArrayList<FeatureRequest>();
                    shardGroups.put(Integer.valueOf(request.shardId), group);
                }
                group.add(request);
            }

            HashMap<Integer, Feature> loaded = new HashMap<Integer, Feature>(Math.max(16, chunk.size() * 2));
            try {
                for (java.util.Map.Entry<Integer, ArrayList<FeatureRequest>> entry : shardGroups.entrySet()) {
                    ArrayList<FeatureRequest> group = entry.getValue();
                    int[] offsets = new int[group.size()];
                    for (int i = 0; i < group.size(); i++) {
                        offsets[i] = group.get(i).offsetInShard;
                    }
                    RowBatch batch = reader.loadRows(entry.getKey().intValue(), offsets);
                    for (int i = 0; i < group.size(); i++) {
                        loaded.put(
                                Integer.valueOf(group.get(i).featureId),
                                new Feature(batch.values[i], batch.valid[i])
                        );
                    }
                }
            } catch (SQLException e) {
                throw new IllegalStateException("failed to load scalar feature batch", e);
            }

            for (FeatureRequest request : chunk) {
                Feature feature = loaded.get(Integer.valueOf(request.featureId));
                if (feature == null) {
                    throw new IllegalStateException("missing loaded feature for feature_id=" + request.featureId);
                }
                currentBatch.add(buildFeatureValues(request, resolvedSamples, feature));
            }
            requestIndex = end;
        }
    }

    private static final Comparator<FeatureRequest> FEATURE_REQUEST_COMPARATOR = new Comparator<FeatureRequest>() {
        @Override
        public int compare(FeatureRequest a, FeatureRequest b) {
            int byShard = Integer.compare(a.shardId, b.shardId);
            if (byShard != 0) {
                return byShard;
            }
            int byOffset = Integer.compare(a.offsetInShard, b.offsetInShard);
            if (byOffset != 0) {
                return byOffset;
            }
            return Integer.compare(a.featureId, b.featureId);
        }
    };

    private static final class FeatureRequest {
        final int featureId;
        final String featureKey;
        final int shardId;
        final int offsetInShard;

        FeatureRequest(int featureId, String featureKey, int shardId, int offsetInShard) {
            this.featureId = featureId;
            this.featureKey = featureKey;
            this.shardId = shardId;
            this.offsetInShard = offsetInShard;
        }
    }

    private static final class ResolvedSamples {
        final long[] sampleIds;
        final String[] sampleKeys;

        ResolvedSamples(long[] sampleIds, String[] sampleKeys) {
            this.sampleIds = sampleIds;
            this.sampleKeys = sampleKeys;
        }
    }
}
