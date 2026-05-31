package fs.config;

/**
 * Array custom binary shard materialize 설정입니다.
 *
 * <p>bundle manifest에서 custom binary shard를 만들 때 쓰는 low-level 설정입니다.
 * 직접 ingestion builder를 쓰는 경우에는 보통 {@link ArrayBinaryBuildOptions}를 사용하고,
 * 이 클래스는 bundle-to-shard 변환을 직접 호출할 때 사용합니다.</p>
 */
public class ArrayShardConfig {
    /** feature 하나를 sample 축으로 자르는 block 크기입니다. */
    public int samplesPerBlock = 16;

    /** shard 하나의 목표 크기(byte)입니다. nShards가 0이면 이 값을 기준으로 자동 분할합니다. */
    public long targetShardBytes = 32L * 1024L * 1024L;

    /** 명시적인 shard 개수입니다. 0이면 targetShardBytes 기준으로 자동 분할합니다. */
    public int nShards = 0;
}
