import argparse

from fs.array.storage import build_array_feature_locator_index, load_array_shard_manifest, shard_file_path


def main():
    """CLI entry point for inspecting array feature locator records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-id", type=int, required=True)
    args = ap.parse_args()

    manifest = load_array_shard_manifest(args.manifest)
    blocks = build_array_feature_locator_index(manifest.locator_path).get(int(args.feature_id), [])
    if not blocks:
        print(f"NOT_FOUND\tfeature_id={args.feature_id}")
        return
    for loc in blocks:
        print(
            f"feature_id={loc.feature_id}\tblock_id={loc.block_id}\tshard_id={loc.shard_id}"
            f"\trow_in_shard={loc.row_in_shard}\tsample_row_start={loc.sample_row_start}"
            f"\tsample_row_end={loc.sample_row_end}\tshard_path={shard_file_path(manifest.shard_path, loc.shard_id)}"
        )


if __name__ == "__main__":
    main()
