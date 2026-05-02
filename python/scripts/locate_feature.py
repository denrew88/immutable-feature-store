import argparse

import polars as pl

from fs.scalar.parquet_storage import build_feature_locator_index, load_manifest, shard_file_path


def main():
    """CLI entry point for inspecting scalar feature locator entries."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-id", type=int, required=True)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    index = build_feature_locator_index(manifest.feature_locator_path)
    loc = index.get(int(args.feature_id))
    if loc is None:
        print(f"NOT_FOUND\tfeature_id={args.feature_id}")
        return
    shard_id, offset_in_shard = loc
    feature_key = None
    feature_meta_path = str(getattr(manifest, "feature_meta_path", "") or "")
    feature_key_col = str(getattr(manifest, "feature_key_col", "feature_key"))
    if feature_meta_path:
        feature_meta_df = pl.read_parquet(feature_meta_path, columns=[feature_key_col])
        if 0 <= int(args.feature_id) < int(feature_meta_df.height):
            feature_key = feature_meta_df[feature_key_col][int(args.feature_id)]
    extra = ""
    locator_df = pl.read_parquet(manifest.feature_locator_path)
    cols = set(locator_df.columns)
    if "r2y" in cols and "n_y_overlap" in cols:
        row = locator_df.filter(locator_df["feature_id"] == int(args.feature_id))
        if row.height > 0:
            extra = f"\tr2y={float(row['r2y'][0])}\tn_y_overlap={int(row['n_y_overlap'][0])}"
    if feature_key is not None:
        extra = f"\tfeature_key={feature_key}{extra}"
    print(
        f"feature_id={args.feature_id}\tshard_id={shard_id}\toffset_in_shard={offset_in_shard}"
        f"{extra}\tshard_path={shard_file_path(manifest.shard_path, shard_id)}"
    )


if __name__ == "__main__":
    main()
