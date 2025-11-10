#!/usr/bin/env bash


if [[ "$(uname)" == "Darwin" ]]; then
    echo "Running on macOS — installing via Homebrew..."
    brew install grep
    GREP_CMD="ggrep"
else
    GREP_CMD="grep"
fi

mkdir -p api/app/models/checkpoints

curl -s https://biomoqa-classifier.s3.text-analytics.ch/ \
| $GREP_CMD -oP '(?<=<Key>)[^<]+' \
| while read -r file; do
    if [[ $file == checkpoints/* ]]; then
        target_file="api/app/models/checkpoints/${file#checkpoints/}"
        mkdir -p "$(dirname "$target_file")"
        echo "Downloading checkpoint: $file -> $target_file"
        wget -O "$target_file" "https://biomoqa-classifier.s3.text-analytics.ch/$file"
    else
        echo "Skipping: $file (not in checkpoints directory)"
    fi
done

echo "Download complete!"
echo "Checkpoints: models/checkpoints/"