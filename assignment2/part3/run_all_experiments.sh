#!/usr/bin/env zsh
# Run all experiments listed in README.md sequentially
# Usage: zsh run_all_experiments.sh

set -e

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  # Ensure conda is initialized in this shell
  if [ -f "$HOME/.zshrc" ]; then
    source "$HOME/.zshrc"
  fi
  conda activate dl2025 || {
    echo "Warning: Could not activate conda env 'dl2025'. Continuing without activation.";
  }
else
  echo "Warning: 'conda' not found in PATH. Running with current Python."
fi

# Move to script directory
SCRIPT_DIR=${0:a:h}
cd "$SCRIPT_DIR"

# Timestamped run
echo "Starting experiments at $(date)"

# List of models to run
models=(
  gcn
  matrix-gcn
  gat
)

for m in $models; do
  echo "\n=== Running model: $m ==="
  python train.py --model "$m" || {
    echo "Experiment for model '$m' failed.";
    exit 1;
  }
  echo "=== Completed: $m ==="
done

echo "\nAll experiments completed at $(date)."

# Hints to view logs
echo "View training curves with:"
echo "  tensorboard --logdir=logs"
