package fs.model.array_sample_parquet;

/**
 * array_sample_parquet v1 build 설정입니다.
 *
 * <p>Java 구현은 sample별 raw parquet를 먼저 만들고 compact 단계에서 raw 파일을
 * 최종 part로 묶습니다. 처음에는 기본값을 쓰거나 {@link #targetPartBytes}만
 * 데이터 규모에 맞게 조정하면 됩니다.</p>
 */
public class ArraySampleParquetBuildOptions {
    /** final sample part 하나의 목표 크기(byte)입니다. part가 너무 많으면 올립니다. */
    public long targetPartBytes = 128L * 1024L * 1024L;

    /** final point part 하나에 넣을 최대 point row 수입니다. */
    public int maxPartRows = 10000000;

    /** final part 하나에 넣을 최대 sample 수입니다. 0이면 sample 수로 제한하지 않습니다. */
    public int maxPartSamples = 0;

    /** parquet compression입니다. 일반 사용은 "zstd", 디버그용 속도 확인은 "none"도 가능합니다. */
    public String compression = "zstd";

    /** sample metadata의 external key column 이름입니다. */
    public String sampleKeyCol = "sample_key";

    /** feature metadata의 external key column 이름입니다. */
    public String featureKeyCol = "feature_key";

    /** DuckDB writer thread 수입니다. 0이면 DuckDB 기본값을 사용합니다. */
    public int duckdbThreads = 0;

    /**
     * Java writer가 DuckDB로 넘기는 Arrow record batch의 최대 point row 수입니다.
     *
     * <p>값이 너무 작으면 batch 수가 늘고, 너무 크면 sample close 시점의 off-heap
     * Arrow buffer 사용량이 커집니다. 0 이하를 넣으면 기본값을 사용합니다.</p>
     */
    public int arrowBatchRows = 262144;
}
