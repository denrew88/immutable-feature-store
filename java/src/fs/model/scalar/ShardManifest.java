package fs.model.scalar;

import java.io.File;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Scalar parquet shard dataset 전체 구성을 설명하는 최상위 manifest다.
 */
public class ShardManifest {
    public final String sampleMetaPath;
    public final String featureMetaPath;
    public final int nSamples;
    public final int nFeatures;
    public final String shardPath;
    public final int nShards;
    public final String featureLocatorPath;
    public final String featureLocatorFormat;
    public final String featureIdType;
    public final String valuesType;
    public final String validType;
    public final String idScheme;
    public final String sampleKeyCol;
    public final String featureKeyCol;
    public final Long targetShardBytes;
    public final Map<String, String> selectionStats;
    public final String statsYCol;

    public ShardManifest(
            String sampleMetaPath,
            String featureMetaPath,
            int nSamples,
            int nFeatures,
            String shardPath,
            int nShards,
            String featureLocatorPath,
            String featureLocatorFormat,
            String featureIdType,
            String valuesType,
            String validType,
            String idScheme,
            String sampleKeyCol,
            String featureKeyCol,
            Long targetShardBytes,
            Map<String, String> selectionStats,
            String statsYCol) {
        this.sampleMetaPath = sampleMetaPath;
        this.featureMetaPath = featureMetaPath;
        this.nSamples = nSamples;
        this.nFeatures = nFeatures;
        this.shardPath = shardPath;
        this.nShards = nShards;
        this.featureLocatorPath = featureLocatorPath;
        this.featureLocatorFormat = featureLocatorFormat;
        this.featureIdType = featureIdType;
        this.valuesType = valuesType;
        this.validType = validType;
        this.idScheme = (idScheme == null || idScheme.isEmpty()) ? "dense_row_ids" : idScheme;
        this.sampleKeyCol = (sampleKeyCol == null) ? "sample_key" : sampleKeyCol;
        this.featureKeyCol = (featureKeyCol == null) ? "feature_key" : featureKeyCol;
        this.targetShardBytes = targetShardBytes;
        this.selectionStats = normalizeSelectionStats(selectionStats);
        this.statsYCol = statsYCol;
    }

    public ShardManifest(
            String sampleMetaPath,
            int nSamples,
            String shardPath,
            int nShards,
            String featureLocatorPath,
            String featureLocatorFormat,
            String featureIdType,
            String valuesType,
            String validType,
            String statsYCol) {
        this(
                sampleMetaPath,
                "",
                nSamples,
                0,
                shardPath,
                nShards,
                featureLocatorPath,
                featureLocatorFormat,
                featureIdType,
                valuesType,
                validType,
                "legacy",
                "sample_key",
                "feature_key",
                null,
                null,
                statsYCol
        );
    }

    public String shardFilePath(int shardId) {
        return new File(shardPath, String.format("shard_%04d.parquet", shardId)).getPath();
    }

    public String selectionStatsPath(String yCol) {
        if (selectionStats.isEmpty() || yCol == null) {
            return "";
        }
        String path = selectionStats.get(yCol);
        return (path == null) ? "" : path;
    }

    private static Map<String, String> normalizeSelectionStats(Map<String, String> selectionStats) {
        if (selectionStats == null || selectionStats.isEmpty()) {
            return Collections.emptyMap();
        }
        LinkedHashMap<String, String> out = new LinkedHashMap<String, String>();
        for (Map.Entry<String, String> entry : selectionStats.entrySet()) {
            out.put(entry.getKey(), entry.getValue());
        }
        return Collections.unmodifiableMap(out);
    }
}
