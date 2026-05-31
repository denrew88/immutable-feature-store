package fs.config;

/**
 * Scalar dense-long shard build 설정입니다.
 *
 * <p>일반 사용자는 보통 {@link #targetShardBytes}, {@link #statsYCols},
 * {@link #denseLongRowGroupFeatures}만 조정하면 됩니다. 나머지 column 이름은
 * metadata/raw parquet schema를 기본값과 다르게 만들었을 때만 바꿉니다.</p>
 */
public class BuildShardConfig {
    /** Legacy shard 개수 override입니다. dense-long 표준 경로에서는 보통 0 그대로 둡니다. */
    public int nShards = 0;

    /** dense-long part 하나의 목표 크기(byte)입니다. part가 너무 많으면 올리고, 파일이 너무 크면 줄입니다. */
    public long targetShardBytes = 32L * 1024L * 1024L;

    /** raw/sample-major 입력의 feature id column 이름입니다. */
    public String featureIdCol = "feature_id";

    /** raw/sample-major 입력의 scalar value column 이름입니다. */
    public String valueCol = "value";

    /** sample id column 이름입니다. dense id 0..N-1이어야 합니다. */
    public String sampleIdCol = "sample_id";

    /** sample metadata의 external key column 이름입니다. key 조회가 필요하면 이 column이 있어야 합니다. */
    public String sampleKeyCol = "sample_key";

    /** feature metadata의 external key column 이름입니다. key 조회가 필요하면 이 column이 있어야 합니다. */
    public String featureKeyCol = "feature_key";

    /** feature metadata parquet 경로입니다. builder session에서는 보통 openSession 인자로 채워집니다. */
    public String featureMetaPath = "";

    /** sample-major manifest/table에서 raw sample parquet path를 담는 column 이름입니다. */
    public String pathCol = "sample_path";

    /** {@link #statsYCols}가 비어 있을 때 selection stats를 만들 target column입니다. */
    public String yCol = "y";

    /** build 중 selection_stats/<y>.parquet를 만들 target column 목록입니다. null이면 {@link #yCol} 하나를 씁니다. */
    public java.util.List<String> statsYCols = null;

    /** value column의 logical type 이름입니다. 현재 dense-long 표준은 FLOAT64입니다. */
    public String valuesType = "FLOAT64";

    /** mask column의 logical type 이름입니다. 현재 dense-long 표준은 UINT8입니다. */
    public String validType = "UINT8";

    /** parquet row group 하나에 묶을 feature 수입니다. 기본 128은 파일 크기와 조회 속도의 균형값입니다. */
    public int denseLongRowGroupFeatures = 128;

    /** part 하나에 넣을 feature 수를 직접 고정합니다. 0이면 targetShardBytes 기준으로 자동 계산합니다. */
    public int denseLongPartFeatures = 0;
}
