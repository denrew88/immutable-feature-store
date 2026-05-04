import argparse

from fs.array.binary_storage import load_array_binary_shard_manifest


def main():
    """Locate the binary shard and block range for one feature id."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-id", type=int, required=True)
    args = ap.parse_args()

    manifest = load_array_binary_shard_manifest(args.manifest)
    feature_id = int(args.feature_id)
    if feature_id < 0 or feature_id >= int(manifest.n_features):
        print(f"NOT_FOUND\tfeature_id={feature_id}")
        return

    for shard in manifest.shards:
        if int(shard.feature_id_start) <= feature_id <= int(shard.feature_id_end):
            local_feature = feature_id - int(shard.feature_id_start)
            record_index_start = local_feature * int(manifest.blocks_per_feature)
            record_index_end = record_index_start + int(manifest.blocks_per_feature) - 1
            print(
                f"feature_id={feature_id}\tshard_id={shard.shard_id}"
                f"\tfeature_id_start={shard.feature_id_start}\tfeature_id_end={shard.feature_id_end}"
                f"\trecord_index_start={record_index_start}\trecord_index_end={record_index_end}"
            )
            return

    print(f"NOT_FOUND\tfeature_id={feature_id}")


if __name__ == "__main__":
    main()
