#!/usr/bin/env bash
# Download model checkpoints from S3.
# Usage:
#   ./download_checkpoints.sh         # download v1 (checkpoints/)
#   ./download_checkpoints.sh v2      # download v2 (checkpoints_v2/)
#   ./download_checkpoints.sh all     # download both

set -euo pipefail

if [[ "$(uname)" == "Darwin" ]]; then
    brew install grep 2>/dev/null || true
    GREP_CMD="ggrep"
else
    GREP_CMD="grep"
fi

S3_BUCKET="https://biomoqa-classifier.s3.text-analytics.ch"
VERSION="${1:-v1}"

download_version() {
    local s3_prefix="$1"
    local local_dir="$2"

    echo "Downloading ${s3_prefix} -> ${local_dir}/"
    mkdir -p "$local_dir"

    curl -s "$S3_BUCKET/" \
    | $GREP_CMD -oP '(?<=<Key>)[^<]+' \
    | while read -r file; do
        if [[ $file == ${s3_prefix}/* ]]; then
            target_file="${local_dir}/${file#${s3_prefix}/}"
            mkdir -p "$(dirname "$target_file")"
            echo "  $file"
            wget -q -O "$target_file" "${S3_BUCKET}/$file"
        fi
    done
}

case "$VERSION" in
    v1)
        download_version "checkpoints" "model/checkpoints"
        ;;
    v2)
        download_version "checkpoints_v2" "model/checkpoints_v2"
        ;;
    all)
        download_version "checkpoints" "model/checkpoints"
        download_version "checkpoints_v2" "model/checkpoints_v2"
        ;;
    *)
        echo "Usage: $0 [v1|v2|all]"
        exit 1
        ;;
esac

echo "Download complete!"
