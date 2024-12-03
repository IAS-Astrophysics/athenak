#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <nurates base directory>"
  exit 1
fi

NURATES_SRC_DIR="$1"

# Copy to current directory of shell script
ATHENA_DEST_DIR=$(dirname "$(readlink -f "$0")")

find "$NURATES_SRC_DIR/include" -type f -exec cp {} "$ATHENA_DEST_DIR" \;
find "$NURATES_SRC_DIR/src/integration" -type f -exec cp {} "$ATHENA_DEST_DIR" \;

echo "Files copied to $ATHENA_DEST_DIR"
