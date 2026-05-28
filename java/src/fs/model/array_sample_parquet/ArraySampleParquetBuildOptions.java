package fs.model.array_sample_parquet;

/**
 * sample-major Parquet array dataset 생성 옵션이다.
 *
 * <p>part 크기는 sample 개수가 아니라 추정 payload byte 기준으로 결정한다.
 * 단, part commit은 sample 경계에서만 수행된다. 따라서 중간 실패 후에는
 * {@link ArraySampleParquetBuildSessionStatus#nextExpectedSampleId}부터 다시 넣으면 된다.
 */
public class ArraySampleParquetBuildOptions {
    public long targetPartBytes = 128L * 1024L * 1024L;
    public int maxPartRows = 100000;
    public int maxPartSamples = 0;
    public String compression = "zstd";
    public String sampleKeyCol = "sample_key";
    public String featureKeyCol = "feature_key";
}
