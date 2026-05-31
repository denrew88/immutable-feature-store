package fs.config;

/**
 * Array custom binary direct builder의 public 설정입니다.
 *
 * <p>sample context로 trace를 직접 넣고 마지막에 custom binary shard를 만들 때 사용합니다.
 * 처음에는 {@link #samplesPerBlock}, {@link #targetShardMb}, {@link #codec} 정도만
 * 지정하면 됩니다.</p>
 */
public class ArrayBinaryBuildOptions {
    /** shard 하나의 목표 크기(MB)입니다. nShards가 null이면 이 값을 기준으로 feature partition을 자동 계산합니다. */
    public int targetShardMb = 32;

    /** feature 하나를 sample 축으로 자르는 block 크기입니다. 작을수록 부분 조회 낭비가 줄고 index overhead가 늘어납니다. */
    public int samplesPerBlock = 16;

    /** 명시적인 shard 개수입니다. null이면 targetShardMb 기준으로 자동 분할합니다. */
    public Integer nShards = null;

    /** block payload codec입니다. 현재 유지보수와 속도 균형상 "none"을 기본 권장값으로 둡니다. */
    public String codec = "none";

    /** codec이 "zstd"일 때만 적용되는 압축 level입니다. */
    public int zstdLevel = 3;

    /** sample metadata의 external key column 이름입니다. */
    public String sampleKeyCol = "sample_key";

    /** feature metadata의 external key column 이름입니다. */
    public String featureKeyCol = "feature_key";
}
