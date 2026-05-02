package fs.config;

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
