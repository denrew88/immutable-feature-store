package fs.io.scalar;

import fs.io.array.ArraySampleIdIndex;

import java.sql.SQLException;

/**
 * Scalar sample id와 key 사이를 변환하는 얇은 인덱스 wrapper다.
 */
public final class SampleIdIndex {
    private final ArraySampleIdIndex delegate;

    private SampleIdIndex(ArraySampleIdIndex delegate) {
        this.delegate = delegate;
    }

    public static SampleIdIndex load(String sampleMetaPath) throws SQLException {
        return new SampleIdIndex(ArraySampleIdIndex.load(sampleMetaPath));
    }

    public static SampleIdIndex load(String sampleMetaPath, String sampleKeyCol) throws SQLException {
        return new SampleIdIndex(ArraySampleIdIndex.load(sampleMetaPath, sampleKeyCol));
    }

    public Long findSampleId(long sampleId) {
        return delegate.findSampleId(sampleId);
    }

    public Long findSampleIdByKey(String sampleKey) {
        return delegate.findSampleIdByKey(sampleKey);
    }

    public String keyForId(long sampleId) {
        return delegate.sampleKey(sampleId);
    }
}
