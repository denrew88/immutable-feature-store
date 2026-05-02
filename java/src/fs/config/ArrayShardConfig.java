package fs.config;

public class ArrayShardConfig {
    public int samplesPerBlock = 16;
    public long targetShardBytes = 32L * 1024L * 1024L;
    public int nShards = 0;
}
