#!/bin/bash

# Script to run CNN experiments for question 1.1
# Runs experiments with different net_type and conv_type combinations
# Usage: ./run_experiments.sh [n_parallel_jobs]
# Example: ./run_experiments.sh 4  # Run 4 experiments in parallel

# Enable strict error handling
set -euo pipefail

# Get number of parallel jobs from argument, default to 1
N_PARALLEL=${1:-1}

# Validate N_PARALLEL is a positive integer
if ! [[ "$N_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Argument must be a positive integer" >&2
    echo "Usage: $0 [n_parallel_jobs]" >&2
    exit 1
fi

SCRIPT_DIR="/Users/a/w/uvadlc_practicals_2025/assignment2/part1/exercise1.1"
LOG_FILE="experiment_results.log"
STATUS_FILE="experiment_status.txt"
PROGRESS_FILE="experiment_progress.txt"

# Initialize counters
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# Change to script directory and verify
if ! cd "$SCRIPT_DIR"; then
    echo "Error: Failed to change to directory $SCRIPT_DIR" >&2
    exit 1
fi

# Check if net.py exists
if [[ ! -f "net.py" ]]; then
    echo "Error: net.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

# Clear previous log and status files
> $LOG_FILE
> $STATUS_FILE
> $PROGRESS_FILE

# Define conv types to test
CONV_TYPES=("valid" "replicate" "reflect" "circular" "sconv" "fconv")

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#CONV_TYPES[@]} * 2))  # 2 net types

echo "================================" | tee -a $LOG_FILE
echo "CNN Experiments for Question 1.1" | tee -a $LOG_FILE
echo "Running with $N_PARALLEL parallel jobs" | tee -a $LOG_FILE
echo "Total experiments: $TOTAL_EXPERIMENTS" | tee -a $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Cleanup function for temporary files
cleanup() {
    local exit_code=$?
    rm -f temp_*.log 2>/dev/null || true
    exit $exit_code
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Function to run a single experiment
run_experiment() {
    local net_type=$1
    local conv_type=$2
    local experiment_id="${net_type}_${conv_type}"
    # Use PID to make temp files unique across parallel processes
    local temp_log="temp_${net_type}_${conv_type}_$$.log"
    
    local start_time=$(date +%s)
    local start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Update status: started
    echo "[${start_timestamp}] STARTED: ${experiment_id}" >> "$STATUS_FILE"
    
    {
        echo "----------------------------------------"
        echo "[${start_timestamp}] Running: $net_type with conv_type=$conv_type"
        echo "----------------------------------------"
        
        # Run experiment and capture exit code
        local exit_code=0
        if python net.py --net_type="$net_type" --conv_type="$conv_type" 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            local end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo ""
            echo "[${end_timestamp}] ✓ Completed: $net_type with conv_type=$conv_type (duration: ${duration}s)"
            echo ""
        else
            exit_code=$?
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            local end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo ""
            echo "[${end_timestamp}] ✗ FAILED: $net_type with conv_type=$conv_type (exit code: $exit_code, duration: ${duration}s)" >&2
            echo ""
        fi
    } > "$temp_log" 2>&1
    
    # Append to main log file
    cat "$temp_log" >> "$LOG_FILE"
    
    # Update status: completed or failed
    local end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local duration=$(($(date +%s) - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        echo "[${end_timestamp}] ✓ SUCCESS: ${experiment_id} (${duration}s)" >> "$STATUS_FILE"
    else
        echo "[${end_timestamp}] ✗ FAILED: ${experiment_id} (${duration}s)" >> "$STATUS_FILE"
    fi
    
    rm -f "$temp_log"
}

# Function to display progress
show_progress() {
    local completed=$(grep -c "SUCCESS\|FAILED" "$STATUS_FILE" 2>/dev/null || echo 0)
    local failed=$(grep -c "FAILED" "$STATUS_FILE" 2>/dev/null || echo 0)
    local success=$((completed - failed))
    local running=$(grep -c "STARTED" "$STATUS_FILE" 2>/dev/null || echo 0)
    running=$((running - completed))
    
    printf "\r[%s] Progress: %d/%d completed (%d success, %d failed) | %d running" \
        "$(date '+%H:%M:%S')" "$completed" "$TOTAL_EXPERIMENTS" "$success" "$failed" "$running"
}

export -f run_experiment
export -f show_progress
export LOG_FILE
export STATUS_FILE
export PROGRESS_FILE
export TOTAL_EXPERIMENTS
export SCRIPT_DIR

# Part (a) - Net1 experiments
echo "========================================" | tee -a $LOG_FILE
echo "Part (a): Running Net1 experiments" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

NET1_START=$(date +%s)
for conv_type in "${CONV_TYPES[@]}"; do
    # Run experiments in parallel with max N_PARALLEL jobs
    while [[ $(jobs -r | wc -l) -ge $N_PARALLEL ]]; do
        show_progress
        sleep 0.5
    done
    
    run_experiment "Net1" "$conv_type" &
    show_progress
done

# Wait for all Net1 experiments to complete with progress updates
echo ""
echo "Waiting for all Net1 experiments to complete..."
while [[ $(jobs -r | wc -l) -gt 0 ]]; do
    show_progress
    sleep 1
done
wait
NET1_END=$(date +%s)
NET1_DURATION=$((NET1_END - NET1_START))

show_progress
echo ""
echo ""
echo "All Net1 experiments completed in ${NET1_DURATION}s" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Part (c) - Net2 experiments
echo "========================================" | tee -a $LOG_FILE
echo "Part (c): Running Net2 experiments" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

NET2_START=$(date +%s)
for conv_type in "${CONV_TYPES[@]}"; do
    # Run experiments in parallel with max N_PARALLEL jobs
    while [[ $(jobs -r | wc -l) -ge $N_PARALLEL ]]; do
        show_progress
        sleep 0.5
    done
    
    run_experiment "Net2" "$conv_type" &
    show_progress
done

# Wait for all Net2 experiments to complete with progress updates
echo ""
echo "Waiting for all Net2 experiments to complete..."
while [[ $(jobs -r | wc -l) -gt 0 ]]; do
    show_progress
    sleep 1
done
wait
NET2_END=$(date +%s)
NET2_DURATION=$((NET2_END - NET2_START))

show_progress
echo ""
echo ""
echo "All Net2 experiments completed in ${NET2_DURATION}s" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Calculate final statistics
TOTAL_SUCCESS=$(grep -c "✓ SUCCESS" "$STATUS_FILE" 2>/dev/null || echo 0)
TOTAL_FAILED=$(grep -c "✗ FAILED" "$STATUS_FILE" 2>/dev/null || echo 0)
TOTAL_DURATION=$((NET1_DURATION + NET2_DURATION))

echo ""
echo "========================================" | tee -a $LOG_FILE
echo "All experiments completed!" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Summary:" | tee -a $LOG_FILE
echo "  Total experiments: $TOTAL_EXPERIMENTS" | tee -a $LOG_FILE
echo "  Successful: $TOTAL_SUCCESS" | tee -a $LOG_FILE
echo "  Failed: $TOTAL_FAILED" | tee -a $LOG_FILE
echo "  Total duration: ${TOTAL_DURATION}s" | tee -a $LOG_FILE
echo "  Net1 duration: ${NET1_DURATION}s" | tee -a $LOG_FILE
echo "  Net2 duration: ${NET2_DURATION}s" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Results saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "Status log saved to: $STATUS_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE