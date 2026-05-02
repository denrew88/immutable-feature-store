# Scalar Parquet Shard Format

This document describes the current scalar feature shard format used in this
repository.

The format is intentionally simpler than the array binary v3 format:

- final serving artifacts are still Parquet-based
- ids are dense row ids
- the artifact is standalone and relocatable as one folder
- feature selection uses sidecar stats files under `selection_stats/`

## 1. Artifact Layout

A scalar shard dataset normally looks like this:

```text
scalar_shard_dataset/
  shard_manifest.json
  sample_meta.parquet
  feature_meta.parquet
  feature_locator.parquet
  selection_stats/
    y.parquet
    y_alt.parquet
    ...
  feature_shards/
    shard_0000.parquet
    shard_0001.parquet
    ...
```

Meaning:

- `shard_manifest.json`
  - top-level metadata
  - relative paths to every dataset component
- `sample_meta.parquet`
  - authoritative sample ordering
- `feature_meta.parquet`
  - authoritative feature ordering
- `feature_locator.parquet`
  - `feature_id -> (shard_id, offset_in_shard)` lookup table
- `selection_stats/<y>.parquet`
  - precomputed candidate stats for a specific target column
- `feature_shards/shard_XXXX.parquet`
  - actual scalar feature rows

All paths written into the manifest are relative to the manifest location.
The whole folder can be moved as one unit.

## 2. Dense Id Rules

Scalar shards use dense ids throughout:

- `sample_id == row index in sample_meta.parquet`
- `feature_id == row index in feature_meta.parquet`

That means:

- row 0 of `sample_meta.parquet` is `sample_id = 0`
- row 17 of `feature_meta.parquet` is `feature_id = 17`

External systems should treat:

- `sample_key`
- `feature_key`

as the stable identifiers.

Dense ids are implementation ids optimized for storage and lookup.

## 3. `sample_meta.parquet`

`sample_meta.parquet` defines the sample axis.

Recommended columns:

- `sample_id`
- `sample_key`
- one or more target columns such as `y`, `y_alt`

Additional user metadata is allowed, for example:

- `split`
- `patient_id`
- `visit_id`
- `group`

Rules:

- if `sample_id` exists, it must equal dense row order `0..n_samples-1`
- `sample_key`, when present, should be non-null and unique

Important:

- the **final shard artifact does not depend on `sample_path`**
- lookup and selection use `sample_meta.parquet`, `feature_locator.parquet`,
  `feature_shards/`, and `selection_stats/`

## 4. `feature_meta.parquet`

`feature_meta.parquet` defines the feature axis.

Recommended columns:

- `feature_id`
- `feature_key`

Additional columns are free-form, for example:

- `group`
- `display_name`
- `modality`
- `feature_kind`

Rules:

- if `feature_id` exists, it must equal dense row order `0..n_features-1`
- `feature_key`, when present, should be non-null and unique

## 5. Feature Shard Row Schema

Each row in a scalar feature shard represents one feature.

Current row schema:

- `feature_id: Int32`
- `value_len: Int32`
- `values_blob: Binary`
- `valid_blob: Binary`

### 5.1 `value_len`

In the current dense scalar format:

- `value_len == n_samples`

for every feature row.

### 5.2 `values_blob`

Physical format:

- little-endian float64 buffer

Length:

- `value_len * 8 bytes`

Decode:

```python
np.frombuffer(values_blob, dtype="<f8", count=value_len)
```

### 5.3 `valid_blob`

Physical format:

- uint8 mask buffer

Length:

- `value_len bytes`

Decode:

```python
np.frombuffer(valid_blob, dtype=np.uint8, count=value_len)
```

Meaning:

- `1` = present
- `0` = missing

## 6. `feature_locator.parquet`

`feature_locator.parquet` is the serving-time lookup table.

Current columns:

- `feature_id: Int32`
- `global_rank: Int32`
- `shard_id: Int32`
- `offset_in_shard: Int32`

Meaning:

- `feature_id`
  - dense feature id
- `global_rank`
  - dense global feature order
- `shard_id`
  - which shard file contains the feature row
- `offset_in_shard`
  - which row inside that shard file

Lookup flow for `feature_id = 123`:

1. read locator row for `feature_id = 123`
2. get `shard_id` and `offset_in_shard`
3. open `feature_shards/shard_XXXX.parquet`
4. read row `offset_in_shard`
5. decode `values_blob` and `valid_blob`

Legacy note:

- older artifacts may also store `r2y` and `n_y_overlap` in the locator
- new artifacts keep candidate stats in `selection_stats/*.parquet` instead

## 7. `selection_stats/`

Precomputed selection stats are stored per target column.

Example:

```text
selection_stats/
  y.parquet
  y_alt.parquet
```

Each stats file contains:

- `feature_id`
- `shard_id`
- `offset_in_shard`
- `r2y`
- `n_y_overlap`

The manifest stores a mapping:

```json
"selection_stats": {
  "y": "selection_stats/y.parquet",
  "y_alt": "selection_stats/y_alt.parquet"
}
```

If the requested `y_col` exists in that mapping, selection can use the
precomputed fast path.

## 8. `shard_manifest.json`

Typical example:

```json
{
  "sample_meta_path": "sample_meta.parquet",
  "feature_meta_path": "feature_meta.parquet",
  "n_samples": 5000,
  "n_features": 20000,
  "shard_path": "feature_shards",
  "n_shards": 8,
  "feature_locator_path": "feature_locator.parquet",
  "feature_locator_format": "parquet_v1",
  "feature_id_dtype": "INT32",
  "values_dtype": "blob_float64_le_len",
  "valid_dtype": "blob_uint8_len",
  "id_scheme": "dense_row_ids",
  "sample_key_col": "sample_key",
  "feature_key_col": "feature_key",
  "target_shard_bytes": 33554432,
  "selection_stats": {
    "y": "selection_stats/y.parquet",
    "y_alt": "selection_stats/y_alt.parquet"
  }
}
```

Important fields:

- `sample_meta_path`
  - relative path to `sample_meta.parquet`
- `feature_meta_path`
  - relative path to `feature_meta.parquet`
- `n_samples`, `n_features`
  - dense axis sizes
- `shard_path`
  - `feature_shards/`
- `feature_locator_path`
  - `feature_locator.parquet`
- `id_scheme`
  - currently `dense_row_ids`
- `selection_stats`
  - precomputed target stats sidecars

Legacy compatibility:

- older manifests may still expose `stats_y_col`
- new artifacts should prefer the `selection_stats` mapping

## 9. Builder Workflow

### 9.1 Final artifact build

The scalar builder produces the final shard artifact from an intermediate
sample-major stage.

### 9.2 Visible sample-major stage

The public builder exposes the intermediate stage explicitly.

Python:

- `finish_sample_major()`
- `build_shards(keep_sample_major=False)`

Java:

- `finishSampleMajor()`
- `buildShards(keepSampleMajor)`

### 9.3 Intermediate stage format

The current builder no longer writes one Parquet file per sample.

Instead, it writes bundled sample-major files:

```text
sample_major_stage/
  sample_meta.parquet
  feature_meta.parquet
  sample_major_manifest.json
  sample_bundles/
    bundle_000000.parquet
    bundle_000001.parquet
    ...
```

Each bundle row is a simple long-format scalar row:

- `sample_id`
- `feature_id`
- `value`

This reduces file count and improves build performance compared to the older
file-per-sample layout.

### 9.4 Keeping or deleting the intermediate stage

Default:

- `keep_sample_major = false`

In that mode the final standalone artifact remains, but the intermediate
sample-major stage is deleted.

If `keep_sample_major = true`, the stage remains on disk for debugging or for
selection fallback workflows that need to rescan intermediate sample-major data.

## 10. Selection Flow

### 10.1 Fast path

If the requested `y_col` exists in `manifest.selection_stats`:

1. load `selection_stats/<y>.parquet`
2. filter by `r2y` and `n_y_overlap`
3. use the shard reader only for the reduced candidate set

### 10.2 Fallback path

If the requested `y_col` does not exist in `manifest.selection_stats`:

1. load target values from `sample_meta.parquet`
2. rescan shard rows
3. recompute candidate stats on the fly

This fallback does not require `sample_path` in the final artifact.

## 11. End-to-End Lookup Example

Suppose:

- `feature_id = 140`
- `sample_id = 118`

Lookup:

1. find `feature_id = 140` in `feature_locator.parquet`
2. get, for example:
   - `shard_id = 2`
   - `offset_in_shard = 17`
3. read row 17 from `feature_shards/shard_0002.parquet`
4. decode:
   - `values_blob -> float64[n_samples]`
   - `valid_blob -> uint8[n_samples]`
5. inspect position `118`
6. if `valid_blob[118] == 1`, return `values_blob[118]`
7. otherwise return missing

Key-based lookup adds one extra step:

1. `feature_key -> feature_id`
2. `sample_key -> sample_id`
3. then follow the same path as above

## 12. Public APIs

### Python

- `ScalarDatasetBuilder`
- `write_sample_meta(...)`
- `write_feature_meta(...)`
- `open_shard(...)`
- `select_features(...)`

### Java

- `fs.io.ScalarDatasetBuilder`
- `fs.io.ScalarFeatureShards`
- `fs.io.ScalarShardDataset`

## 13. Validation Checklist

Reader and builder implementations should validate at least:

### Metadata

- `sample_meta.parquet` row count == `n_samples`
- `feature_meta.parquet` row count == `n_features`
- `sample_id` matches row order
- `feature_id` matches row order
- `sample_key` and `feature_key`, if present, are unique

### Locator

- `feature_locator.parquet` row count == `n_features`
- every `feature_id` is unique
- `shard_id` is in range
- `offset_in_shard` is valid for that shard

### Feature rows

- `feature_id` matches locator entry
- `value_len == n_samples`
- `values_blob.length == value_len * 8`
- `valid_blob.length == value_len`

### Selection stats

- every configured stats file matches the manifest key
- `feature_id`, `shard_id`, `offset_in_shard` agree with locator

## 14. Summary

The current scalar format is:

- Parquet-based
- standalone
- dense-id based
- key-aware
- optimized for lookup and selection

The final serving artifact does not depend on `sample_path`.
The public direct-ingestion builders use a visible bundled sample-major stage,
then materialize the final standalone shard artifact.
