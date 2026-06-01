import argparse

import polars as pl

from fs._package_sources import ensure_package_source

ensure_package_source("scalar_feature_shard")

from scalar_feature_shard.dense_long import load_dense_long_manifest


def main():
    """CLI entry point for inspecting dense-long scalar feature locator entries."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-id", type=int, required=True)
    args = ap.parse_args()

    manifest = load_dense_long_manifest(args.manifest)
    locator_df = pl.read_parquet(manifest.feature_locator_path)
    row = locator_df.filter(pl.col("feature_id") == int(args.feature_id))
    if row.height == 0:
        print(f"NOT_FOUND\tfeature_id={args.feature_id}")
        return

    part_id = int(row["part_id"][0])
    offset_in_part = int(row["offset_in_part"][0])
    part_path = next((part.path for part in manifest.parts if int(part.part_id) == part_id), None)

    feature_key = None
    feature_key_col = str(getattr(manifest, "feature_key_col", "") or "")
    if feature_key_col:
        feature_meta_df = pl.read_parquet(manifest.feature_meta_path, columns=[feature_key_col])
        if 0 <= int(args.feature_id) < int(feature_meta_df.height):
            feature_key = feature_meta_df[feature_key_col][int(args.feature_id)]

    extra = "" if feature_key is None else f"\tfeature_key={feature_key}"
    print(
        f"feature_id={args.feature_id}\tpart_id={part_id}\toffset_in_part={offset_in_part}"
        f"{extra}\tpart_path={part_path}"
    )


if __name__ == "__main__":
    main()
