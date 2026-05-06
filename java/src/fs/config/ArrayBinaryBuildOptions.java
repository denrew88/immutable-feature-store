package fs.config;

/**
 * Python의 ArrayBinaryBuildOptions에 대응하는 자바 public build 옵션이다.
 *
 * <p>이 객체는 direct-ingestion builder와 facade가 자주 받는 고수준 옵션만 묶는다.
 * 내부 spill planner나 low-level bundle flush 설정은 별도 config에 남겨 둔다.
 */
public class ArrayBinaryBuildOptions {
    public int targetShardMb = 32;
    public int samplesPerBlock = 16;
    public Integer nShards = null;
    public String codec = "none";
    public int zstdLevel = 3;
    public String sampleKeyCol = "sample_key";
    public String featureKeyCol = "feature_key";
}
