package fs.config;

public class ArraySyntheticConfig {
    public int nSamples = 1000;
    public int nFeatures = 256;
    public int nLatentGroups = 12;
    public double informativeGroupRatio = 0.3;
    public int nLatentForY = 4;
    public double groupSizeMean = 24.0;
    public double noiseFeatureRatio = 0.2;
    public double redundantStrength = 1.25;
    public double noiseScale = 0.15;
    public int minTraceLen = 96;
    public int maxTraceLen = 384;
    public double emptyTraceRate = 0.02;
    public double missingFeatureRate = 0.1;
    public double nonfiniteTraceRate = 0.03;
    public double nonfinitePointRate = 0.02;
    public double timeDuration = 10.0;
    public double timeJitter = 0.25;
    public long sampleIdOffset = 0L;
    public long seed = 0L;
}
