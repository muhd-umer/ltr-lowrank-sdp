#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PROCESSOR="$PROJECT_ROOT/dataset/processor.py"
DATA_DIR="$PROJECT_ROOT/dataset/instances/"
OUTPUT_DIR="$PROJECT_ROOT/dataset/proc"
TIMEOUT_DURATION=900

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

cleanup() {
    echo -e "\nScript interrupted. Exiting gracefully."
    exit 1
}

trap cleanup SIGINT

if [ ! -f "$PROCESSOR" ]; then
    echo "Error: Processor script not found at $PROCESSOR"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Processing SDPA files to PyG graphs"
echo "  Input:  $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

total=0
success=0
failed=0
skipped=0
timeout_count=0

for subfolder in "$DATA_DIR"*/; do
    if [ -d "$subfolder" ]; then
        subfolder_name=$(basename "$subfolder")
        
        for file in "$subfolder"*.dat-s; do
            if [ -f "$file" ]; then
                total=$((total + 1))
                problem_name=$(basename "$file" .dat-s)
                output_file="$OUTPUT_DIR/${problem_name}.pt"

                # skip if already processed
                if [ -f "$output_file" ]; then
                    echo "[$total] skipping (exists): $problem_name"
                    skipped=$((skipped + 1))
                    continue
                fi

                echo "[$total] processing: $subfolder_name/$problem_name"

                timeout $TIMEOUT_DURATION python "$PROCESSOR" --input "$file" --output "$output_file" 2>&1 | while read line; do
                    echo "    $line"
                done
                exit_code=${PIPESTATUS[0]}

                if [ $exit_code -eq 0 ]; then
                    echo "    [p] success"
                    success=$((success + 1))
                    echo ""
                elif [ $exit_code -eq 124 ]; then
                    echo "    [t] timeout after ${TIMEOUT_DURATION}s"
                    timeout_count=$((timeout_count + 1))
                    rm -f "$output_file"
                    echo ""
                else
                    echo "    [x] failed; exit code $exit_code"
                    failed=$((failed + 1))
                    echo ""
                fi
            fi
        done
    fi
done

echo ""
echo "<summary>"
echo "  total problems:    $total"
echo "  successful:        $success"
echo "  skipped:           $skipped"
echo "  timeout:           $timeout_count"
echo "  failed:            $failed"
