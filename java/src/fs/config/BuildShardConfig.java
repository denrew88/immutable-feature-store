package fs.config;

/**
 * Scalar shard 빌드 단계에서 shard 목표 크기와 selection 통계 생성을 제어하는 설정이다.
 */
public class BuildShardConfig {
    public int nShards = 0;
    public long targetShardBytes = 32L * 1024L * 1024L;
    public String featureIdCol = "feature_id";
    public String valueCol = "value";
    public String sampleIdCol = "sample_id";
    public String sampleKeyCol = "sample_key";
    public String featureKeyCol = "feature_key";
    public String featureMetaPath = "";
    public String pathCol = "sample_path";
    public String yCol = "y";
    public java.util.List<String> statsYCols = null;
    public String valuesType = "FLOAT64";
    public String validType = "UINT8";
}
