package fs.config;

/**
 * Array binary shard 빌드 단계에서 shard 크기와 block 배치를 제어하는 설정이다.
 */
public class ArrayShardConfig {
    public int samplesPerBlock = 16;
    public long targetShardBytes = 32L * 1024L * 1024L;
    public int nShards = 0;
}
