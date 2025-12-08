import os
import re
import argparse
import pickle
from decimal import Decimal

import numpy as np
from solver.solver_utils import SOLVER_CLASSES

from utils.utils import TASKS

def get_best_obj_and_time(files):
    obj = 0
    time = 0
    length = len(files)
    for file_path in files:
        if "test.log" in file_path:
           length -= 1
           continue

    for file_path in files:
        if "test.log" in file_path:
           continue
        with open(file_path, "r", encoding="utf-8") as f:
            log_text = f.read()

        time_match = re.search(r"in\s+([0-9.]+)\s+seconds", log_text)
        run_time = float(time_match.group(1)) if time_match else None

        obj_match = re.search(r"Best objective\s+([0-9.eE+-]+)", log_text)
        best_obj = float(Decimal(obj_match.group(1))) if obj_match else None
        obj += best_obj / length
        time += run_time / length
    print("test instance numbers: ", length)
    print("average obj: ", obj) 
    print("average time: ", time)

def get_log_files(file_dir, file_suffix):
  target_files = []
  for entry in os.scandir(file_dir):
      if entry.is_file():
        file_name = entry.name
        if file_name.endswith(file_suffix):
          target_files.append(entry.path)
  return sorted(target_files)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIP Solver with GNN-based Prediction")
    
    exp_group = parser.add_argument_group("Experiment Settings")
    exp_group.add_argument("--task", type=str, choices=TASKS, default='CA')
    exp_group.add_argument("--fix_strategy", default="pas")
    exp_group.add_argument("--method_type", default='RoME', choices=['RoME', 'PS'],
                       help="Training method type (default: %(default)s)")
    exp_group.add_argument("--time_flag", type=str, default='20251207_232349')
    exp_group.add_argument("--gnn_type", default="moe", choices=["gcn", "moe"],
                           help="GNN architecture type (default: %(default)s)")
    exp_group.add_argument("--num_shared_experts", type=int, default=1)
    exp_group.add_argument("--num_dedicate_experts", type=int, default=5)

    solver_group = parser.add_argument_group("Solver Settings")
    solver_group.add_argument("--solver", choices=SOLVER_CLASSES.keys(), default="gurobi")
    solver_group.add_argument("--max_time", type=int, default=1000)
    solver_group.add_argument("--k0", type=int, default=400)
    solver_group.add_argument("--k1", type=int, default=0)
    solver_group.add_argument("--delta", type=int, default=60)
 
    sys_group = parser.add_argument_group("System Settings")
    sys_group.add_argument("--log_dir", default="./test_logs/")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Construct save_name format consistent with multi_test.py
    if args.gnn_type == 'moe':
        save_name = f'{args.time_flag}_{args.gnn_type}_shared_{args.num_shared_experts}_dedicate_{args.num_dedicate_experts}'
    else:
        save_name = f'{args.time_flag}_{args.gnn_type}'
    
    # Build log directory path: test_logs/method_type/task/test/
    log_dir = os.path.join(args.log_dir, args.method_type, args.task, f"{save_name}")
    
    print(f"Looking for logs in: {log_dir}")
    print(f"Save name pattern: {save_name}")
    
    if not os.path.exists(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        exit(1)
    
    # Get all matching log files
    log_files = []
    for entry in os.scandir(log_dir):
        if entry.is_file() and entry.name.endswith('.log'):
            # Match log files with specific pattern
            if save_name in entry.name and 'instance_' in entry.name:
                log_files.append(entry.path)
    
    log_files = sorted(log_files)

    if len(log_files) == 0:
        print("No matching log files found.")
        exit(1)
    
    # Calculate average objective function value
    get_best_obj_and_time(log_files)