#!/bin/bash

LORADS_EXECUTABLE="./lorads/src/build/LoRADS_v_2_0_1-alpha"
DATA_DIR="./dataset/instances/"
TIMEOUT_DURATION=300
LOG_DIR="./dataset/logs/"
JSON_DIR="./dataset/sol_json/"

cleanup() {
    echo -e "\nscript interrupted. exiting..."
    exit 1
}

trap cleanup SIGINT

if [ ! -f "$LORADS_EXECUTABLE" ]; then
    echo "[error]: LoRADS executable not found at $LORADS_EXECUTABLE"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "[error]: data directory not found at $DATA_DIR"
    exit 1
fi

get_lorads_params() {
    local problem_name="$1"
    local n="$2"

    local phase1_tol="1e-3"
    local heuristic_factor="1.0"
    local times_log_rank="2.0"
    local rho_max="5000.0"

    if [[ "$problem_name" =~ ^[Gg][0-9] ]] || \
       [[ "$problem_name" =~ ^maxcut ]] || \
       [[ "$problem_name" =~ ^mcp ]] || \
       [[ "$problem_name" =~ ^p2p ]] || \
       [[ "$problem_name" =~ ^delaunay ]] || \
       [[ "$problem_name" =~ ^rgg ]] || \
       [[ "$problem_name" =~ ^vsp ]] || \
       [[ "$problem_name" =~ ^cs[0-9] ]] || \
       [[ "$problem_name" =~ ^cit ]] || \
       [[ "$problem_name" =~ ^fe_ ]] || \
       [[ "$problem_name" =~ ^amazon ]] || \
       [[ "$problem_name" =~ [0-9]+a$ ]]; then

        if [[ -n "$n" ]] && [[ "$n" -ge 40000 ]]; then
            phase1_tol="1e+1"
            heuristic_factor="100.0"
        else
            phase1_tol="1e-2"
            heuristic_factor="10.0"
        fi

    elif [[ "$problem_name" =~ ^[Mm][Cc]_ ]]; then
        local mc_n=$(echo "$problem_name" | grep -oP '(?<=MC_|mc_)[0-9]+')

        if [[ -n "$mc_n" ]]; then
            if [[ "$mc_n" -ge 10000 ]]; then
                # MC_10000 and larger
                heuristic_factor="2.5"
                times_log_rank="1.0"
            elif [[ "$mc_n" -ge 1000 ]]; then
                # MC_1000 to MC_8000
                heuristic_factor="5.0"
            fi
        fi

    # maxCut with specific patterns (mb, mc suffix)
    elif [[ "$problem_name" =~ _mb$ ]] || [[ "$problem_name" =~ mc$ ]]; then
        phase1_tol="1e-2"
        heuristic_factor="10.0"
    fi

    echo "--phase1Tol $phase1_tol --heuristicFactor $heuristic_factor --timesLogRank $times_log_rank --rhoMax $rho_max --timeSecLimit $TIMEOUT_DURATION"
}

echo "running LoRADS on all problem instances in $DATA_DIR"

total=0
success=0
failed=0

for subfolder in "$DATA_DIR"*/; do
    if [ -d "$subfolder" ]; then
        for file in "$subfolder"*.dat-s; do
            if [ -f "$file" ]; then
                total=$((total + 1))
                problem_name=$(basename "$file" .dat-s)

                params=$(get_lorads_params "$problem_name")

                echo ""
                echo "[$total] processing: $problem_name"
                echo "    parameters: $params"

                log_file="${LOG_DIR}${problem_name}.log"
                json_file="${JSON_DIR}${problem_name}.json"

                "$LORADS_EXECUTABLE" "$file" --logfile "$log_file" --jsonfile "$json_file" $params > /dev/null 2>&1
                exit_code=$?

                if [ $exit_code -eq 0 ]; then
                    echo "    [p] SUCCESS"
                    success=$((success + 1))
                else
                    echo "    [x] FAILED: Exit code $exit_code"
                    failed=$((failed + 1))
                fi
            fi
        done
    fi
done

echo ""
echo "<summary>"
echo "  total problems:    $total"
echo "  successful:        $success"
echo "  failed:            $failed"