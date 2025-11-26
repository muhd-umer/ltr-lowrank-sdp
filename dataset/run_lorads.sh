#!/bin/bash

LORADS_EXECUTABLE="./lorads/src/build/LoRADS_v_2_0_1-alpha"
DATA_DIR="./dataset/inst/"

cleanup() {
    echo -e "\nScript interrupted. Exiting gracefully."
    exit 1
}

trap cleanup SIGINT

if [ ! -f "$LORADS_EXECUTABLE" ]; then
    echo "Error: LoRADS executable not found at $LORADS_EXECUTABLE"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

echo "Running LoRADS on all problem instances in $DATA_DIR"

for file in "$DATA_DIR"/*.dat-s; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        "$LORADS_EXECUTABLE" "$file" || true
    fi
done
