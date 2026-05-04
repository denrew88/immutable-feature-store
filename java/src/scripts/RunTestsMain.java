package scripts;

import fs.config.SelectionConfig;
import fs.config.SyntheticConfig;
import fs.model.synthetic.SyntheticData;
import fs.synth.SyntheticGenerator;
import fs.validate.Validation;

/**
 * Scalar synthetic 생성과 selection 계산의 기본 정합성을 점검하는 테스트 엔트리포인트다.
 */
public class RunTestsMain {
    public static void main(String[] args) throws Exception {
        long seed = getLongArg(args, "--seed", 0L);
        SyntheticConfig cfg = new SyntheticConfig();
        cfg.nSamples = 200;
        cfg.nFeatures = 300;
        cfg.seed = seed;

        SyntheticData data = SyntheticGenerator.generate(cfg);
        double[] y = data.y;
        byte[] yMask = new byte[y.length];
        for (int i = 0; i < y.length; i++) {
            yMask[i] = (byte) (Double.isNaN(y[i]) ? 0 : 1);
        }

        Validation.validateBatchKernel(data.X, data.M, y, yMask, 20, 1e-6);

        SelectionConfig sc = new SelectionConfig();
        sc.yR2Threshold = 0.01;
        sc.minNonNullY = 20;
        sc.ffR2Threshold = 0.9;
        sc.minNonNullPair = 20;
        sc.topM = 30;
        sc.initialCap = 50;
        sc.maxStep = 100;
        sc.batchSize = 64;

        Validation.validateIncrementalVsNaive(data.X, data.M, y, yMask, sc, 64);
        System.out.println("tests passed");
    }

    private static long getLongArg(String[] args, String key, long defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return Long.parseLong(args[i + 1]);
            }
        }
        return defaultValue;
    }
}
