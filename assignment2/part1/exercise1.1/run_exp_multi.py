#!/usr/bin/env python3
"""
Run all experiments for question 1.1 using net_multi.py
Automatically saves logs for all experiments
Usage: python run_exp_multi.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
CONV_TYPES = ["valid", "replicate", "reflect", "circular", "sconv", "fconv"]
NET_TYPES = ["Net1", "Net2"]


def run_experiment(net_type, conv_type):
    """Run a single experiment with logging enabled"""
    experiment_id = f"{net_type}_{conv_type}"
    start_time = datetime.now()
    
    print(f"\n{'='*60}")
    print(f"[{start_time.strftime('%H:%M:%S')}] Starting: {experiment_id}")
    print(f"{'='*60}")
    
    # Run the experiment with --log flag
    try:
        result = subprocess.run(
            [sys.executable, "net_multi.py", 
             f"--net_type={net_type}", 
             f"--conv_type={conv_type}", 
             "--log"],
            cwd=SCRIPT_DIR
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {experiment_id} completed in {duration:.1f}s")
            return True
        else:
            print(f"\n✗ FAILED: {experiment_id} (exit code: {result.returncode}) after {duration:.1f}s")
            return False
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n✗ ERROR: {experiment_id} failed with exception: {e}")
        return False


def main():
    # Check if net_multi.py exists
    net_multi_py = SCRIPT_DIR / "net_multi.py"
    if not net_multi_py.exists():
        print(f"Error: net_multi.py not found in {SCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)
    
    # Print header
    total_experiments = len(CONV_TYPES) * len(NET_TYPES)
    print("="*60)
    print("CNN Experiments for Question 1.1 (Multi-device)")
    print(f"Total experiments: {total_experiments}")
    print("All results will be saved to logs/ directory")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    overall_start = datetime.now()
    results = []
    completed = 0
    
    # Run experiments for each net type
    for net_type in NET_TYPES:
        print(f"\n{'#'*60}")
        print(f"# Part {'(a)' if net_type == 'Net1' else '(c)'}: Running {net_type} experiments ({len(CONV_TYPES)} total)")
        print(f"{'#'*60}")
        
        for conv_type in CONV_TYPES:
            completed += 1
            print(f"\nProgress: {completed}/{total_experiments}")
            
            success = run_experiment(net_type, conv_type)
            results.append({
                'net_type': net_type,
                'conv_type': conv_type,
                'success': success
            })
    
    # Print summary
    overall_duration = (datetime.now() - overall_start).total_seconds()
    total_success = sum(1 for r in results if r['success'])
    total_failed = len(results) - total_success
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Total time: {overall_duration:.1f}s ({overall_duration/60:.1f} min)")
    print("="*60)
    
    # Print experiment results mapping
    print("\nResults saved to logs/ directory:")
    print("Each experiment has its own timestamped folder containing:")
    print("  - config.json: Experiment configuration")
    print("  - metrics.json: Validation and test accuracies")
    print("  - training_curves.png: Training progress visualization")
    print("  - final_accuracies.png: Final accuracy comparison")
    print("  - summary_statistics.png: Summary statistics")
    
    # Print failed experiments if any
    if total_failed > 0:
        print("\n" + "="*60)
        print("FAILED EXPERIMENTS:")
        print("="*60)
        for r in results:
            if not r['success']:
                print(f"  - {r['net_type']}_{r['conv_type']}")
    
    print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with error if any experiments failed
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
