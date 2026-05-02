package fs.io;

import fs.model.Feature;
import fs.model.FeatureLocation;
import fs.model.ScalarFeatureValues;
import fs.model.ScalarValue;
import fs.model.ShardManifest;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public final class ScalarShardDataset implements AutoCloseable {
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
        Integer resolvedFeatureId = featureIds.findFeatureId(featureId);
        if (resolvedFeatureId == null) {
            throw new IllegalArgumentException("unknown feature_id: " + featureId);
        }
        Feature feature = loadFeature(resolvedFeatureId.intValue());
        ArrayList<ScalarValue> out = new ArrayList<ScalarValue>(sampleIds.length);
        for (long sampleId : sampleIds) {
            out.add(valueForSample(feature, resolvedFeatureId.intValue(), sampleId));
        }
        return new ScalarFeatureValues(
                resolvedFeatureId.intValue(),
                featureIds.keyForId(resolvedFeatureId.intValue()),
                out
        );
    }

    public ScalarFeatureValues getValuesBySampleKeys(int featureId, String[] sampleKeys) throws SQLException {
        long[] resolved = new long[sampleKeys.length];
        for (int i = 0; i < sampleKeys.length; i++) {
            Long sampleId = sampleIds.findSampleIdByKey(sampleKeys[i]);
            if (sampleId == null) {
                throw new IllegalArgumentException("unknown sample_key: " + sampleKeys[i]);
            }
            resolved[i] = sampleId.longValue();
        }
        return getValues(featureId, resolved);
    }

    public ScalarFeatureValues getValuesByKeys(String featureKey, String[] sampleKeys) throws SQLException {
        Integer featureId = featureIds.findFeatureIdByKey(featureKey);
        if (featureId == null) {
            throw new IllegalArgumentException("unknown feature_key: " + featureKey);
        }
        return getValuesBySampleKeys(featureId.intValue(), sampleKeys);
    }

    public ScalarValue getValue(int featureId, long sampleId) throws SQLException {
        return getValues(featureId, new long[]{sampleId}).values.get(0);
    }

    public ScalarValue getValueByKey(String featureKey, String sampleKey) throws SQLException {
        return getValuesByKeys(featureKey, new String[]{sampleKey}).values.get(0);
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

    private ScalarValue valueForSample(Feature feature, int featureId, long sampleId) {
        Long resolvedSampleId = sampleIds.findSampleId(sampleId);
        if (resolvedSampleId == null) {
            throw new IllegalArgumentException("unknown sample_id: " + sampleId);
        }
        int idx = (int) resolvedSampleId.longValue();
        boolean present = idx < feature.valid.length && feature.valid[idx] != 0;
        Double value = present ? Double.valueOf(feature.values[idx]) : null;
        return new ScalarValue(
                sampleId,
                sampleIds.keyForId(sampleId),
                present,
                value
        );
    }
}
