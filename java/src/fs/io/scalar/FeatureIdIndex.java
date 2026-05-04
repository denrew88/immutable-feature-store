package fs.io.scalar;

import fs.io.array.ArrayFeatureIdIndex;

import java.sql.SQLException;

/**
 * Scalar feature id와 key 사이를 변환하는 얇은 인덱스 wrapper다.
 */
public final class FeatureIdIndex {
    private final ArrayFeatureIdIndex delegate;

    private FeatureIdIndex(ArrayFeatureIdIndex delegate) {
        this.delegate = delegate;
    }

    public static FeatureIdIndex load(String featureMetaPath) throws SQLException {
        return new FeatureIdIndex(ArrayFeatureIdIndex.load(featureMetaPath));
    }

    public static FeatureIdIndex load(String featureMetaPath, String featureKeyCol) throws SQLException {
        return new FeatureIdIndex(ArrayFeatureIdIndex.load(featureMetaPath, featureKeyCol));
    }

    public Integer findFeatureId(int featureId) {
        return delegate.findFeatureId(featureId);
    }

    public Integer findFeatureIdByKey(String featureKey) {
        return delegate.findFeatureIdByKey(featureKey);
    }

    public String keyForId(int featureId) {
        return delegate.featureKey(featureId);
    }
}
