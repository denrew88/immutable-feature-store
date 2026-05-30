package fs.io.array_sample_parquet;

import fs.io.common.DuckDBUtils;
import fs.model.array_sample_parquet.ArraySampleParquetBuildOptions;
import fs.model.array_sample_parquet.ArraySampleParquetPart;
import fs.model.common.PointColumnSpec;

import java.io.File;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

/**
 * sample별 raw parquet 파일들을 최종 sample part parquet으로 묶는 compactor.
 *
 * <p>Java가 row를 다시 읽어서 쓰지 않는다. raw parquet 목록만 DuckDB에 넘기고,
 * DuckDB가 {@code read_parquet -> COPY TO parquet} 경로로 scan/write를 처리한다.
 * raw file 목록은 sample_id 순서이고 각 raw file은 이미 {@code (sample_id, feature_id, point_idx)}
 * 순서이므로 compact 단계에서는 별도 SQL sort를 하지 않는다.</p>
 */
final class ArraySampleParquetCompactor {
    private ArraySampleParquetCompactor() {
    }

    static List<ArraySampleParquetPart> compact(
            File outDir,
            File samplePartsDir,
            File traceIndexPartsDir,
            List<RawSampleRecord> records,
            List<PointColumnSpec> pointSchema,
            ArraySampleParquetBuildOptions options,
            boolean overwrite) throws Exception {
        if (overwrite) {
            deleteChildren(samplePartsDir);
            deleteChildren(traceIndexPartsDir);
        }
        ensureDir(samplePartsDir);
        ensureDir(traceIndexPartsDir);

        ArrayList<CompactPlan> plans = planParts(records, options);
        ArrayList<ArraySampleParquetPart> parts = new ArrayList<ArraySampleParquetPart>();
        for (int partId = 0; partId < plans.size(); partId++) {
            CompactPlan plan = plans.get(partId);
            File pointFinal = new File(samplePartsDir, String.format("part_%06d.parquet", partId)).getAbsoluteFile();
            File traceFinal = new File(traceIndexPartsDir, String.format("part_%06d.parquet", partId)).getAbsoluteFile();
            File pointTmp = new File(pointFinal.getAbsolutePath() + ".tmp");
            File traceTmp = new File(traceFinal.getAbsolutePath() + ".tmp");
            deleteQuietly(pointTmp);
            deleteQuietly(traceTmp);

            try (Connection conn = DuckDBUtils.connect(null);
                 Statement st = conn.createStatement()) {
                ArraySampleParquetDuckDB.configure(conn, outDir, options);
                st.execute(copyPointsSql(plan.rawPointPaths, pointSchema, pointTmp.getAbsolutePath(), options.compression));
                st.execute(copyTraceIndexSql(plan.rawTraceIndexPaths, traceTmp.getAbsolutePath(), options.compression));
            } catch (Exception e) {
                deleteQuietly(pointTmp);
                deleteQuietly(traceTmp);
                throw e;
            }

            moveTmpToFinal(pointTmp, pointFinal);
            moveTmpToFinal(traceTmp, traceFinal);
            parts.add(new ArraySampleParquetPart(
                    partId,
                    pointFinal.getAbsolutePath(),
                    traceFinal.getAbsolutePath(),
                    plan.firstSampleId,
                    plan.lastSampleId,
                    plan.sampleCount,
                    plan.traceCount,
                    plan.rowCount,
                    pointFinal.length(),
                    traceFinal.length()));
        }
        return parts;
    }

    private static ArrayList<CompactPlan> planParts(List<RawSampleRecord> records, ArraySampleParquetBuildOptions options) {
        ArrayList<CompactPlan> plans = new ArrayList<CompactPlan>();
        CompactPlan current = new CompactPlan();
        long targetBytes = options == null ? 128L * 1024L * 1024L : options.targetPartBytes;
        int maxRows = options == null ? 10000000 : options.maxPartRows;
        int maxSamples = options == null ? 0 : options.maxPartSamples;

        for (RawSampleRecord record : records) {
            boolean wouldExceed = current.sampleCount > 0
                    && ((maxRows > 0 && current.rowCount + record.rowCount > maxRows)
                    || (targetBytes > 0L && current.byteSize + record.totalByteSize() > targetBytes)
                    || (maxSamples > 0 && current.sampleCount + 1 > maxSamples));
            if (wouldExceed) {
                plans.add(current);
                current = new CompactPlan();
            }
            current.add(record);
        }
        if (current.sampleCount > 0) {
            plans.add(current);
        }
        return plans;
    }

    private static String copyPointsSql(List<String> rawPaths, List<PointColumnSpec> pointSchema, String outPath, String compression) {
        StringBuilder sb = new StringBuilder();
        sb.append("COPY (SELECT sample_id, feature_id, point_idx");
        for (PointColumnSpec spec : pointSchema) {
            sb.append(", ").append(DuckDBUtils.quoteIdentifier(spec.name));
        }
        sb.append(" FROM read_parquet(").append(ArraySampleParquetDuckDB.pathListLiteral(rawPaths)).append("))");
        sb.append(" TO ").append(DuckDBUtils.quotePath(outPath)).append(" ");
        sb.append(ArraySampleParquetDuckDB.parquetCopyOptions(compression));
        return sb.toString();
    }

    private static String copyTraceIndexSql(List<String> rawPaths, String outPath, String compression) {
        return "COPY (SELECT sample_id, feature_id, trace_len FROM read_parquet("
                + ArraySampleParquetDuckDB.pathListLiteral(rawPaths)
                + ")) TO "
                + DuckDBUtils.quotePath(outPath) + " "
                + ArraySampleParquetDuckDB.parquetCopyOptions(compression);
    }

    private static void moveTmpToFinal(File tmp, File finalPath) throws IOException {
        try {
            Files.move(tmp.toPath(), finalPath.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp.toPath(), finalPath.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void ensureDir(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("failed to create directory: " + dir.getAbsolutePath());
        }
    }

    private static void deleteChildren(File dir) {
        if (!dir.exists()) {
            return;
        }
        File[] children = dir.listFiles();
        if (children == null) {
            return;
        }
        for (File child : children) {
            deleteRecursively(child);
        }
    }

    private static void deleteRecursively(File file) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (file.exists() && !file.delete()) {
            throw new IllegalStateException("failed to delete: " + file.getAbsolutePath());
        }
    }

    private static void deleteQuietly(File file) {
        if (file.exists() && !file.delete()) {
            // best-effort cleanup
        }
    }

    static final class RawSampleRecord {
        final long sampleId;
        final String sampleKey;
        final String pointPath;
        final String traceIndexPath;
        final int traceCount;
        final int rowCount;
        final long byteSize;
        final long traceIndexByteSize;

        RawSampleRecord(
                long sampleId,
                String sampleKey,
                String pointPath,
                String traceIndexPath,
                int traceCount,
                int rowCount,
                long byteSize,
                long traceIndexByteSize) {
            this.sampleId = sampleId;
            this.sampleKey = sampleKey;
            this.pointPath = pointPath;
            this.traceIndexPath = traceIndexPath;
            this.traceCount = traceCount;
            this.rowCount = rowCount;
            this.byteSize = byteSize;
            this.traceIndexByteSize = traceIndexByteSize;
        }

        long totalByteSize() {
            return byteSize + traceIndexByteSize;
        }
    }

    private static final class CompactPlan {
        final ArrayList<String> rawPointPaths = new ArrayList<String>();
        final ArrayList<String> rawTraceIndexPaths = new ArrayList<String>();
        long firstSampleId = -1L;
        long lastSampleId = -1L;
        int sampleCount;
        int traceCount;
        int rowCount;
        long byteSize;

        void add(RawSampleRecord record) {
            if (sampleCount == 0) {
                firstSampleId = record.sampleId;
            }
            lastSampleId = record.sampleId;
            sampleCount++;
            traceCount += record.traceCount;
            rowCount += record.rowCount;
            byteSize += record.totalByteSize();
            rawPointPaths.add(record.pointPath);
            rawTraceIndexPaths.add(record.traceIndexPath);
        }
    }
}
