import os
import argparse
import subprocess
import itertools
from datetime import datetime


def run_experiment(data_dir, output_dir, config):
    """Run a single experiment with the given configuration"""
    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "train.py",
        "--data-dir", data_dir,
        "--output-dir", experiment_dir
    ]
    
    # Add configuration parameters
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Log configuration
    config_str = " ".join([f"{k}={v}" for k, v in config.items()])
    log_file = os.path.join(experiment_dir, "log.txt")
    
    with open(log_file, "w") as f:
        f.write(f"Configuration: {config_str}\n\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
    
    # Run the experiment
    print(f"Running experiment with config: {config_str}")
    
    try:
        # Append output to log file
        with open(log_file, "a") as f:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end="")  # Print to console
                f.write(line)  # Write to log file
            
            process.wait()
            
            if process.returncode != 0:
                print(f"Experiment failed with return code {process.returncode}")
                f.write(f"\nExperiment failed with return code {process.returncode}")
            else:
                print("Experiment completed successfully")
                f.write("\nExperiment completed successfully")
    
    except Exception as e:
        print(f"Error running experiment: {e}")
        with open(log_file, "a") as f:
            f.write(f"\nError running experiment: {e}")


def run_grid_search(data_dir, output_dir, param_grid):
    """Run a grid search over the parameter space"""
    # Create parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Running grid search with {len(configurations)} configurations")
    
    # Create grid search directory
    grid_search_dir = os.path.join(output_dir, f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Run each configuration
    for i, config in enumerate(configurations):
        print(f"\nRunning configuration {i+1}/{len(configurations)}")
        run_experiment(data_dir, grid_search_dir, config)


def main():
    parser = argparse.ArgumentParser(description='Run experiments for Modified ResNet CIFAR-10')
    parser.add_argument('--data-dir', default='/scratch/tf2387/deep-learning-spring-2025-project-1/extracted/cifar-10-batches-py',
                        help='path to CIFAR-10 data')
    parser.add_argument('--output-dir', default='/home/tf2387/7123pj_zzp/outputs',
                        help='path to save outputs')
    parser.add_argument('--mode', choices=['single', 'grid'], default='single',
                        help='run a