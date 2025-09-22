#!/bin/bash

# --- Experiment Configuration ---
SCENARIOS=("education" "medication" "fertilizer")
MODELS=("hierarchical_mcfr" "mcfrnet" "causal_forest")
NSAMPLES=(100 500 1000 3000 8000)
KAPPAS=(0.5 2.0 5.0)
SEEDS=$(seq 110 120)

# --- NEW: Create a timestamped parent directory for this entire run ---
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
PARENT_DIR="results/${TIMESTAMP}"
echo "--- Starting new experiment run. All results will be saved in: ${PARENT_DIR} ---"

# Check for a --dry_run flag
DRY_RUN_FLAG=""
if [ "$1" == "--dry_run" ]; then
    DRY_RUN_FLAG="--dry_run"
    echo "--- Starting DRY RUN ---"
fi

# --- Main Experiment Loop ---
for scenario in "${SCENARIOS[@]}"; do
    for model in "${MODELS[@]}"; do
        for n in "${NSAMPLES[@]}"; do
            for kappa in "${KAPPAS[@]}"; do
                for seed in $SEEDS; do
                    
                    # --- UPDATED: Define directory structure within the timestamped parent folder ---
                    OUTPUT_DIR="${PARENT_DIR}/${scenario}/${model}/n${n}_k${kappa}/seed${seed}"
                    mkdir -p "$OUTPUT_DIR"
                    
                    # Construct the command
                    COMMAND="python mcfr.py --scenario $scenario --model_type $model --n_samples $n --kappa $kappa --seed $seed --output_dir "$OUTPUT_DIR" $DRY_RUN_FLAG"
                    
                    echo "---------------------------------------------------------"
                    echo "Running: $COMMAND"
                    echo "Output will be saved in: $OUTPUT_DIR"
                    echo "---------------------------------------------------------"

                    # Execute the command and redirect output to a log file
                    $COMMAND > "$OUTPUT_DIR/run.log" 2>&1 &
                    
                done
                # Wait for all seeds of a given condition to finish before starting the next
                wait
            done
        done
    done
done

echo "--- All experiments launched! ---"