#!/usr/bin/env python3
"""
Script to run CNN experiments for question 1.1
Runs experiments with different net_type and conv_type combinations in parallel
Usage: python run_experiments.py [n_parallel_jobs]
Example: python run_experiments.py 4  # Run 4 experiments in parallel
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
import argparse

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "experiment_results.log"
STATUS_FILE = SCRIPT_DIR / "experiment_status.txt"
CONV_TYPES = ["valid", "replicate", "reflect", "circular", "sconv", "fconv"]
NET_TYPES = ["Net1", "Net2"]

# Thread-safe file writing
file_lock = Lock()

# Global verbose flag
VERBOSE = False


def log_message(message, to_file=True, to_console=True):
    """Write message to log file and/or console"""
    if to_console:
        print(message)
    if to_file:
        with file_lock:
            with open(LOG_FILE, 'a') as f:
                f.write(message + '\n')


def update_status(message):
    """Update status file"""
    with file_lock:
        with open(STATUS_FILE, 'a') as f:
            f.write(message + '\n')


def run_experiment(net_type, conv_type):
    """Run a single experiment"""
    experiment_id = f"{net_type}_{conv_type}"
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Update status: started
    update_status(f"[{start_timestamp}] STARTED: {experiment_id}")
    
    # Build log message
    log_lines = []
    log_lines.append("----------------------------------------")
    log_lines.append(f"[{start_timestamp}] Running: {net_type} with conv_type={conv_type}")
    log_lines.append("----------------------------------------")
    
    try:
        # Run the experiment
        if VERBOSE:
            # Run with real-time output
            process = subprocess.Popen(
                [sys.executable, "net.py", f"--net_type={net_type}", f"--conv_type={conv_type}", "--log"],
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                # Print with task ID prefix
                print(f"Output from task {experiment_id}: {line}", flush=True)
            
            process.wait(timeout=3600)
            result_stdout = '\n'.join(output_lines)
            result_stderr = ''
            result_returncode = process.returncode
        else:
            # Run with captured output
            result = subprocess.run(
                [sys.executable, "net.py", f"--net_type={net_type}", f"--conv_type={conv_type}", "--log"],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            result_stdout = result.stdout
            result_stderr = result.stderr
            result_returncode = result.returncode
        
        # Capture output
        log_lines.append(result_stdout)
        if result_stderr:
            log_lines.append(result_stderr)
        
        duration = int(time.time() - start_time)
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if result_returncode == 0:
            log_lines.append("")
            log_lines.append(f"[{end_timestamp}] ✓ Completed: {net_type} with conv_type={conv_type} (duration: {duration}s)")
            log_lines.append("")
            
            # Write to log file
            with file_lock:
                with open(LOG_FILE, 'a') as f:
                    f.write('\n'.join(log_lines) + '\n')
            
            # Update status
            update_status(f"[{end_timestamp}] ✓ SUCCESS: {experiment_id} ({duration}s)")
            return {"success": True, "experiment_id": experiment_id, "duration": duration}
        else:
            log_lines.append("")
            log_lines.append(f"[{end_timestamp}] ✗ FAILED: {net_type} with conv_type={conv_type} (exit code: {result_returncode}, duration: {duration}s)")
            log_lines.append("")
            
            # Write to log file
            with file_lock:
                with open(LOG_FILE, 'a') as f:
                    f.write('\n'.join(log_lines) + '\n')
            
            # Update status
            update_status(f"[{end_timestamp}] ✗ FAILED: {experiment_id} ({duration}s)")
            return {"success": False, "experiment_id": experiment_id, "duration": duration}
            
    except subprocess.TimeoutExpired:
        duration = int(time.time() - start_time)
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_lines.append("")
        log_lines.append(f"[{end_timestamp}] ✗ TIMEOUT: {net_type} with conv_type={conv_type} (duration: {duration}s)")
        log_lines.append("")
        
        with file_lock:
            with open(LOG_FILE, 'a') as f:
                f.write('\n'.join(log_lines) + '\n')
        
        update_status(f"[{end_timestamp}] ✗ TIMEOUT: {experiment_id} ({duration}s)")
        return {"success": False, "experiment_id": experiment_id, "duration": duration}
    
    except Exception as e:
        duration = int(time.time() - start_time)
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_lines.append("")
        log_lines.append(f"[{end_timestamp}] ✗ ERROR: {net_type} with conv_type={conv_type}: {str(e)}")
        log_lines.append("")
        
        with file_lock:
            with open(LOG_FILE, 'a') as f:
                f.write('\n'.join(log_lines) + '\n')
        
        update_status(f"[{end_timestamp}] ✗ ERROR: {experiment_id} ({duration}s)")
        return {"success": False, "experiment_id": experiment_id, "duration": duration}


def show_progress(completed, total, success, failed, running=0):
    """Display progress information"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\r[{timestamp}] Progress: {completed}/{total} completed ({success} success, {failed} failed) | {running} running", 
          end='', flush=True)


def run_experiments_for_net(net_type, n_parallel):
    """Run all experiments for a specific net type"""
    experiments = [(net_type, conv_type) for conv_type in CONV_TYPES]
    results = []
    completed = 0
    success = 0
    failed = 0
    
    print(f"\nStarting {len(experiments)} experiments for {net_type}...")
    
    with ProcessPoolExecutor(max_workers=n_parallel) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_experiment, net, conv): (net, conv)
            for net, conv in experiments
        }
        
        running = len(future_to_exp)
        show_progress(completed, len(experiments), success, failed, running)
        
        # Process completed experiments
        for future in as_completed(future_to_exp):
            result = future.result()
            results.append(result)
            completed += 1
            running -= 1
            
            if result["success"]:
                success += 1
            else:
                failed += 1
            
            show_progress(completed, len(experiments), success, failed, running)
    
    print()  # New line after progress
    return results


def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(
        description='Run CNN experiments with different net_type and conv_type combinations'
    )
    parser.add_argument(
        'n_parallel',
        type=int,
        nargs='?',
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Display real-time output from each task'
    )
    args = parser.parse_args()
    
    n_parallel = args.n_parallel
    VERBOSE = args.verbose
    
    # Validate n_parallel
    if n_parallel < 1:
        print("Error: Number of parallel jobs must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    # Check if net.py exists
    net_py = SCRIPT_DIR / "net.py"
    if not net_py.exists():
        print(f"Error: net.py not found in {SCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)
    
    # Clear previous log and status files
    LOG_FILE.write_text('')
    STATUS_FILE.write_text('')
    
    # Calculate total experiments
    total_experiments = len(CONV_TYPES) * len(NET_TYPES)
    
    # Print header
    header = [
        "================================",
        "CNN Experiments for Question 1.1",
        f"Running with {n_parallel} parallel jobs",
        f"Total experiments: {total_experiments}",
        f"Date: {datetime.now().strftime('%c')}",
        "================================",
        ""
    ]
    for line in header:
        log_message(line)
    
    overall_start = time.time()
    all_results = []
    
    # Run experiments for each net type
    for net_type in NET_TYPES:
        section_header = [
            "========================================",
            f"Running {net_type} experiments",
            "========================================",
            ""
        ]
        for line in section_header:
            log_message(line)
        
        net_start = time.time()
        results = run_experiments_for_net(net_type, n_parallel)
        all_results.extend(results)
        net_duration = int(time.time() - net_start)
        
        completion_msg = f"\nAll {net_type} experiments completed in {net_duration}s"
        log_message(completion_msg)
        log_message("")
    
    # Calculate final statistics
    overall_duration = int(time.time() - overall_start)
    total_success = sum(1 for r in all_results if r["success"])
    total_failed = len(all_results) - total_success
    
    # Print summary
    summary = [
        "",
        "========================================",
        "All experiments completed!",
        "",
        "Summary:",
        f"  Total experiments: {total_experiments}",
        f"  Successful: {total_success}",
        f"  Failed: {total_failed}",
        f"  Total duration: {overall_duration}s",
        "",
        f"Results saved to: {LOG_FILE.name}",
        f"Status log saved to: {STATUS_FILE.name}",
        "========================================"
    ]
    for line in summary:
        log_message(line)
    
    # Exit with error if any experiments failed
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
