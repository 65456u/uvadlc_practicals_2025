#!/bin/bash
# Run all experiments for question 1.1 and save terminal output
# This script runs run_exp_multi.py and saves all output to a file

cd /Users/a/w/uvadlc_practicals_2025/assignment2/part1/exercise1.1

# Activate conda environment
conda activate dl2025

# Run experiments and save output to both file and terminal
echo "Starting experiments at $(date)"
echo "Output will be saved to terminal_output_$(date +%Y%m%d_%H%M%S).txt"

python run_exp_multi.py 2>&1 | tee terminal_output_$(date +%Y%m%d_%H%M%S).txt

echo "Experiments completed at $(date)"
