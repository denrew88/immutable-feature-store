package fs.math;

/**
 * Selection 단계에서 쓰는 Pearson 상관계수 계산 helper다.
 */
public class Pearson {
    public static class PairwiseResult {
        public final double r2;
        public final int n;

        public PairwiseResult(double r2, int n) {
            this.r2 = r2;
            this.n = n;
        }
    }

    public static class BatchResult {
        public final double[] r2;
        public final int[] n;

        public BatchResult(double[] r2, int[] n) {
            this.r2 = r2;
            this.n = n;
        }
    }

    public static PairwiseResult pairwiseR2(double[] x, byte[] mx, double[] z, byte[] mz, int minNonNull) {
        if (x.length != z.length || mx.length != mz.length || x.length != mx.length) {
            throw new IllegalArgumentException("x/z and mx/mz length must match");
        }
        int n = 0;
        double sumX = 0.0;
        double sumZ = 0.0;
        double sumX2 = 0.0;
        double sumZ2 = 0.0;
        double sumXZ = 0.0;

        for (int i = 0; i < x.length; i++) {
            if (mx[i] == 0 || mz[i] == 0) {
                continue;
            }
            double xv = x[i];
            double zv = z[i];
            if (Double.isNaN(xv) || Double.isNaN(zv)) {
                continue;
            }
            n++;
            sumX += xv;
            sumZ += zv;
            sumX2 += xv * xv;
            sumZ2 += zv * zv;
            sumXZ += xv * zv;
        }

        if (n < Math.max(2, minNonNull)) {
            return new PairwiseResult(0.0, n);
        }

        double mxv = sumX / n;
        double mzv = sumZ / n;
        double varX = sumX2 - n * mxv * mxv;
        double varZ = sumZ2 - n * mzv * mzv;
        if (varX <= 0.0 || varZ <= 0.0) {
            return new PairwiseResult(0.0, n);
        }
        double cov = sumXZ - n * mxv * mzv;
        double r = cov / Math.sqrt(varX * varZ);
        return new PairwiseResult(r * r, n);
    }


    public static BatchResult batchR2OneVsMany(double[] x, byte[] mx, double[][] Z, byte[][] Mz, int minNonNull) {
        int B = Z.length;
        int N = x.length;
        if (mx.length != N) {
            throw new IllegalArgumentException("mx length must match x length");
        }
        double[] r2 = new double[B];
        int[] n = new int[B];

        for (int i = 0; i < B; i++) {
            double[] zi = Z[i];
            byte[] mzi = Mz[i];
            if (zi.length != N || mzi.length != N) {
                throw new IllegalArgumentException("Z/Mz row length must match x length");
            }
            int count = 0;
            double sumX = 0.0;
            double sumZ = 0.0;
            double sumX2 = 0.0;
            double sumZ2 = 0.0;
            double sumXZ = 0.0;

            for (int j = 0; j < N; j++) {
                if (mx[j] == 0 || mzi[j] == 0) {
                    continue;
                }
                double xv = x[j];
                double zv = zi[j];
                if (Double.isNaN(xv) || Double.isNaN(zv)) {
                    continue;
                }
                count++;
                sumX += xv;
                sumZ += zv;
                sumX2 += xv * xv;
                sumZ2 += zv * zv;
                sumXZ += xv * zv;
            }

            n[i] = count;
            if (count < minNonNull) {
                r2[i] = 0.0;
                continue;
            }
            double mxv = sumX / count;
            double mzv = sumZ / count;
            double varX = sumX2 - count * mxv * mxv;
            double varZ = sumZ2 - count * mzv * mzv;
            if (varX <= 0.0 || varZ <= 0.0) {
                r2[i] = 0.0;
                continue;
            }
            double cov = sumXZ - count * mxv * mzv;
            double r = cov / Math.sqrt(varX * varZ);
            r2[i] = r * r;
        }

        return new BatchResult(r2, n);
    }

}
