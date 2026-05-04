package fs.config;

/**
 * Selection 단계에서 후보 수와 상관계수 임계값을 제어하는 설정이다.
 */
public class SelectionConfig {
    public double yR2Threshold = 0.01;
    public int minNonNullY = 200;
    public double ffR2Threshold = 0.9;
    public int minNonNullPair = 200;
    public int topM = 100;
    public int initialCap = 2048;
    public int maxStep = 4096;
    public int batchSize = 1024;
    public int maxGap = 0;
    public int maxCandidates = 0;
}
