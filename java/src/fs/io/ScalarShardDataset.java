package fs.io;

import fs.io.scalar.DuckDBShardReader;
import fs.io.scalar.FeatureIdIndex;
import fs.io.scalar.FeatureLocatorIndex;
import fs.io.scalar.ManifestIO;
import fs.io.scalar.SampleIdIndex;
import fs.model.common.Feature;
import fs.model.common.FeatureLocation;
import fs.model.scalar.RowBatch;
import fs.model.scalar.ScalarFeatureValues;
import fs.model.scalar.ScalarValue;
import fs.model.scalar.ShardManifest;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * scalar shard dataset를 읽는 high-level facade이다.
 *
 * <p>단일 feature 조회와 batched iteration을 모두 제공하고,
 * 외부에서 받은 sample/feature key를 내부 dense id로 해석하는 책임도 이 계층이 맡는다.
 */
public final class ScalarShardDataset implements AutoCloseable {
    private static final int DEFAULT_BATCH_SIZE = 128;

    private final ShardManifest manifest;
    private final FeatureLocatorIndex locator;
    private final SampleIdIndex sampleIds;
    private final FeatureIdIndex featureIds;
    private final DuckDBShardReader reader;

    /**
     * manifest 경로에서 dataset facade를 연다.
     */
    public ScalarShardDataset(String manifestPath) throws Exception {
        this(ManifestIO.read(manifestPath));
    }

    /**
     * 이미 로드한 manifest에서 dataset facade를 연다.
     */
    public ScalarShardDataset(ShardManifest manifest) throws Exception {
        this.manifest = manifest;
        this.locator = FeatureLocatorIndex.load(manifest);
        this.sampleIds = SampleIdIndex.load(manifest.sampleMetaPath, manifest.sampleKeyCol);
        this.featureIds = FeatureIdIndex.load(manifest.featureMetaPath, manifest.featureKeyCol);
        this.reader = new DuckDBShardReader(manifest);
    }

    /**
     * reader가 사용 중인 shard manifest를 반환한다.
     */
    public ShardManifest manifest() {
        return manifest;
    }

    /**
     * dense feature id와 dense sample id 배열을 기준으로 값을 읽는다.
     *
     * <p>즉 입력이 이미 내부 표준 id 체계라고 가정하는 가장 직접적인 조회 경로이다.
     * sample key나 feature key 해석이 필요 없을 때 가장 단순하게 사용한다.
     */
    public ScalarFeatureValues getValues(int featureId, long[] sampleIds) throws SQLException {
        FeatureRequest request = normalizeFeatureRequests(new int[]{featureId}, null, true).get(0);
        ResolvedSamples resolvedSamples = resolveSampleIds(sampleIds);
        Feature feature = loadFeature(request.featureId);
        return buildFeatureValues(request, resolvedSamples, feature);
    }

    /**
     * dense feature id와 sample key 배열을 기준으로 값을 읽는다.
     *
     * <p>{@link #getValues(int, long[])}와 달리 feature는 이미 dense id로 받지만,
     * sample 쪽은 key를 내부 dense sample id로 변환한 뒤 조회한다.
     */
    public ScalarFeatureValues getValuesBySampleKeys(int featureId, String[] sampleKeys) throws SQLException {
        FeatureRequest request = normalizeFeatureRequests(new int[]{featureId}, null, true).get(0);
        ResolvedSamples resolvedSamples = resolveSampleKeys(sampleKeys);
        Feature feature = loadFeature(request.featureId);
        return buildFeatureValues(request, resolvedSamples, feature);
    }

    /**
     * feature key와 sample key 조합으로 값을 읽는다.
     *
     * <p>세 조회 메서드 중 가장 바깥쪽 API이다.
     * feature와 sample을 모두 key로 받아 dense id로 해석한 뒤 조회하므로, 외부 사용자 코드에서 가장 쓰기 쉽다.
     */
    public ScalarFeatureValues getValuesByKeys(String featureKey, String[] sampleKeys) throws SQLException {
        ArrayList<FeatureRequest> requests = normalizeFeatureKeyRequests(new String[]{featureKey}, true);
        ResolvedSamples resolvedSamples = resolveSampleKeys(sampleKeys);
        Feature feature = loadFeature(requests.get(0).featureId);
        return buildFeatureValues(requests.get(0), resolvedSamples, feature);
    }

    /**
     * scalar 값 하나만 읽는다.
     */
    public ScalarValue getValue(int featureId, long sampleId) throws SQLException {
        return getValues(featureId, new long[]{sampleId}).values.get(0);
    }

    /**
     * feature key와 sample key 조합으로 값 하나만 읽는다.
     */
    public ScalarValue getValueByKey(String featureKey, String sampleKey) throws SQLException {
        return getValuesByKeys(featureKey, new String[]{sampleKey}).values.get(0);
    }

    /**
     * 여러 feature를 배치로 읽되, 입력은 dense feature id / dense sample id 기준으로 받는다.
     *
     * <p>{@link #getValues(int, long[])}를 feature 여러 개에 대해 확장한 형태이며,
     * 결과는 {@link ScalarFeatureValues}를 하나씩 순회하는 iterable로 돌려준다.
     */
    public Iterable<ScalarFeatureValues> iterMany(int[] requestedFeatureIds, long[] requestedSampleIds) throws SQLException {
        return iterMany(requestedFeatureIds, requestedSampleIds, DEFAULT_BATCH_SIZE, true);
    }

    /**
     * 여러 feature를 배치로 읽되, 입력은 dense feature id / dense sample id 기준으로 받는다.
     *
     * <p>{@code maintainOrder=true}면 입력 feature 순서를 그대로 유지한다.
     * {@code maintainOrder=false}면 feature를 shard/offset 순으로 재정렬해 locality를 높이고,
     * 반환 순서도 그 재정렬된 순서를 따른다.
     */
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

    /**
     * feature key / sample key 기준으로 batched iteration을 수행한다.
     *
     * <p>{@link #iterMany(int[], long[])}와 목적은 같지만, 입력을 dense id 대신 key로 받는다.
     * 따라서 외부 호출자는 id index를 따로 준비하지 않아도 된다.
     */
    public Iterable<ScalarFeatureValues> iterManyByKey(String[] requestedFeatureKeys, String[] requestedSampleKeys) throws SQLException {
        return iterManyByKey(requestedFeatureKeys, requestedSampleKeys, DEFAULT_BATCH_SIZE, true);
    }

    /**
     * feature key / sample key 기준으로 batched iteration을 수행한다.
     *
     * <p>{@code maintainOrder} 의미는 {@link #iterMany(int[], long[], int, boolean)}와 동일하지만,
     * 정렬과 batching 이전에 먼저 feature key와 sample key를 dense id로 해석한다는 점이 다르다.
     */
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

    /**
     * feature id 요청을 shard/offset 정보가 붙은 내부 request 목록으로 바꾼다.
     */
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

    /**
     * feature key 요청을 dense feature id와 shard 위치가 붙은 내부 request 목록으로 바꾼다.
     */
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

    /**
     * feature batch를 shard별로 묶어 실제 row read를 수행하는 iterator 구현체이다.
     */
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

        /**
         * 다음 feature chunk를 shard별로 묶어 읽고, 현재 batch buffer를 채운다.
         */
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
