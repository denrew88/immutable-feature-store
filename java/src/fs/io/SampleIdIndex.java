package fs.io;

import java.sql.SQLException;

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
