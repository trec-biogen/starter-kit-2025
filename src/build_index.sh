#!/bin/bash
#
# download and index pubmed baseline using lucene-based pyserini
#

URL="http://bionlp.nlm.nih.gov/biogen-2025-document-collection.zip"
ZIP_FILE="biogen-2025-document-collection.zip"
TARGET_DIR="pubmed_baseline_collection_jsonl"

echo "Downloading file..."
wget -O "$ZIP_FILE" "$URL"

TEMP_DIR="temp_extraction_dir"
mkdir -p "$TEMP_DIR"

echo "Unzipping ZIP file..."
unzip -q "$ZIP_FILE" -d "$TEMP_DIR"

EXTRACTED_SUBDIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -d "$EXTRACTED_SUBDIR" ]; then
    echo "Renaming/moving extracted folder to $TARGET_DIR..."
    mv "$EXTRACTED_SUBDIR" "$TARGET_DIR"
else
    echo "No folder found in ZIP; copying files directly to $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    mv "$TEMP_DIR"/* "$TARGET_DIR"/
fi

rm -rf $ZIP_FILE
rm -rf $TEMP_DIR

export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"

/data/guptadk/anaconda3/envs/biogen2025/bin/python -m pyserini.index.lucene \
       --collection JsonCollection \
       --input $TARGET_DIR \
       --index ../data/indexes/pubmed_baseline_collection_jsonl \
       --generator DefaultLuceneDocumentGenerator \
       --threads 25 \
       --storePositions --storeDocvectors --storeRaw

echo "finished!"