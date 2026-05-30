package fs.model.array_sample_parquet;

/**
 * array_sample_parquet v1 build 옵션.
 *
 * <p>Java 구현은 sample별 raw parquet를 먼저 만들고, compact 단계에서 raw 파일들을 최종 part로
 * 묶는다. part 크기는 sample 개수가 아니라 추정 파일 크기와 point row 수를 기준으로 결정한다.</p>
 */
public class ArraySampleParquetBuildOptions {
    public long targetPartBytes = 128L * 1024L * 1024L;
    public int maxPartRows = 10000000;
    public int maxPartSamples = 0;
    public String compression = "zstd";
    public String sampleKeyCol = "sample_key";
    public String featureKeyCol = "feature_key";
    public int duckdbThreads = 0;
    /**
     * Java writer가 DuckDB로 넘기는 Arrow record batch의 최대 point row 수.
     *
     * <p>값이 너무 작으면 DuckDB가 처리해야 하는 batch 수가 늘고, 너무 크면 sample close 시점의
     * off-heap Arrow buffer 사용량이 커진다. 0 이하를 넣으면 기본값을 사용한다.</p>
     */
    public int arrowBatchRows = 262144;
}
