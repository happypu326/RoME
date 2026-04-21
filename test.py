import pickle
import argparse
import random
import os
import logging
import numpy as np
import torch
from datetime import datetime

from typing import Dict, List, Tuple
from model.moe import MoEPolicy
from utils.utils import get_a_new2, TASKS
from solver.solver_utils import SOLVER_CLASSES
from model.gcn import GNNPolicy
from multiprocessing import Process, Queue, cpu_count, set_start_method

def setup_environment(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def configure_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "test.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_pretrained_model(args: argparse.Namespace, model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if args.gnn_type == 'gcn':
        model = GNNPolicy(emb_size=args.emb_size, 
                          constraint_nfeats=args.constraint_nfeats, 
                          edge_nfeats=args.edge_nfeats, 
                          variable_nfeats=args.variable_nfeats).to(device)
    else:
        model = MoEPolicy(emb_size=args.emb_size,
                          constraint_nfeats=args.constraint_nfeats, 
                          edge_nfeats=args.edge_nfeats, 
                          variable_nfeats=args.variable_nfeats, 
                          num_shared_experts=args.num_shared_experts,
                          num_dedicate_experts=args.num_dedicate_experts,
                          top_k=args.top_k,
                          gate_temperature=args.gate_temperature,
                          bias_lr=args.bias_lr,
                          dropout=0.1,
                          use_dro=args.use_dro,
                          eps_wasserstein=args.eps_wasserstein,
                          dro_perturb_type=args.dro_perturb_type).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    return model

def process_single_instance(gnn_type,  ins_path, policy, device):
    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_path)
    constraint_features = c_nodes.cpu()
    mask = torch.isnan(constraint_features)
    constraint_features[mask] = 1
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)
    batch_indices = torch.zeros(v_nodes.shape[0], dtype=torch.long)

    if gnn_type == 'gcn':
        BD = policy(
            constraint_features.to(device),
            edge_indices.to(device),
            edge_features.to(device),
            variable_features.to(device),
            batch_indices.to(device),
        )
        BD = BD.sigmoid().cpu().squeeze()
    else:
        BD, _ = policy(
            constraint_features.to(device),
            edge_indices.to(device),
            edge_features.to(device),
            variable_features.to(device),
            batch_indices.to(device)
        )
        BD = BD.sigmoid().cpu().squeeze()

    #align the variable name betweend the output and the solver
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    scores=[]#get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])

    scores.sort(key=lambda x:x[2],reverse=True)

    scores=[x for x in scores if x[4]=='BINARY']
    return scores

def fix_pas(scores, task):
    hyperparam = {"IP": (60, 30, 70), "MIS": (600, 600, 100), "WA": (20, 200, 100), "CA": (150, 0, 25), "SC" : (300, 0, 100), 
                   "BP": (100, 0, 50), "CFLP": (30, 0, 30), "GISP": (1000, 100, 120), "LB":   (300, 100, 150), "MVC":  (40, 0, 30),
                   "MIRP": (0.6, 0, 0.3), "MMCN": (0.5, 0, 0.3), "NNV": (0.6, 0, 0.3),"OTS": (0.6, 0, 0.3),"SRPN": (0.5, 0, 0.3), 
                   "GC": (0.6, 0, 0.3), "JS": (0.5, 0, 0.3), "MC":(0.5, 0, 0.3), "NF":(0.6, 0, 0.3), "PF":(0.5, 0, 0.3)
                   }
    k0_raw, k1_raw, delta_raw = hyperparam[task]

    N = len(scores)

    def compute_k(k_raw):
        if isinstance(k_raw, float) and 0 < k_raw < 1:
            return int(k_raw * N)
        else:
            return int(k_raw)

    k0 = compute_k(k0_raw)
    k1 = compute_k(k1_raw)
    delta = compute_k(delta_raw)

    k0 = min(k0, N)
    k1 = min(k1, N)
    delta =min(delta, N)

    scores.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(len(scores), k1)):
        scores[i][3] = 1

    scores.sort(key=lambda x: x[2], reverse=False)
    for i in range(min(len(scores), k0)):
        scores[i][3] = 0

    return scores, delta

def fix_soft_confidence(scores, task):
    hyperparam = {"CA": (0.99, 0.12), "IP": (0.93, 0.01), "SC": (0.96, 0.02), "MIS": (0.95, 0.05), "CFLP": (0.98, 0.02),
                 "GISP": (0.95, 0.05), "LB":   (0.99, 0.15), "MVC":  (0.97, 0.02), "BP": (0.95, 0.05),
                 "MIRP":(0.99, 0.20), "MMCN": (0.97, 0.15), "NNV": (0.97, 0.30),"OTS": (0.97, 0.15),"SRPN": (0.97, 0.15),
                 "IIS": (0.97, 0.02)} 
    threshold, alpha = hyperparam[task]
    tot_fix = 0

    for i in range(len(scores)):
        if scores[i][2] > threshold:
            scores[i][3] = 1
            tot_fix += 1

    for i in range(len(scores)):
        if scores[i][2] < 1 - threshold:
            scores[i][3] = 0
            tot_fix += 1

    print(len(scores), tot_fix)
    return scores, tot_fix * alpha

def solve_mps(mps_file, log_dir, save_name, ins_name, scores, task, args):
    log_file = log_dir
    solver = SOLVER_CLASSES[args.solver]()
    solver.hide_output_to_console()

    # read instance
    solver.load_model(mps_file)

    solver.set_aggressive()

    if args.fix_strategy == 'pas':
        scores, delta = fix_pas(scores, task)
    else:
        scores, delta = fix_soft_confidence(scores, task)

    # trust region method implemented by adding constraints
    instance_variables = solver.get_vars()
    instance_variables.sort(key=lambda v: solver.varname(v))

    variables_map = {}
    for v in instance_variables:  # get a dict (variable map), varname:var clasee
        variables_map[solver.varname(v)] = v

    alphas = []

    for i in range(len(scores)):
        tar_var = variables_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue

        tmp_var = solver.create_real_var(name=f'alpha_{tar_var}')
        alphas.append(tmp_var)
        solver.add_constraint(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        solver.add_constraint(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')

    if len(alphas) > 0:
        all_tmp = 0
        for tmp in alphas:
            all_tmp += tmp
        solver.add_constraint(all_tmp <= delta, name="sum_alpha")

    results = solver.solve(means=args.solver, log_file=log_file, time_limit=args.max_time, threads=args.threads)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIP Solver with GNN-based Prediction")
    
    exp_group = parser.add_argument_group("Experiment Settings")
    exp_group.add_argument("--method_type", default="RoME", choices=["PS", "RoME"],help="Testing methodology (default: %(default)s)")
    exp_group.add_argument("--test_problem_type", type=str, choices=TASKS, default='CA',help="Problem type to train on (e.g., IS, WA, IP)")
    exp_group.add_argument("--training_problem_types", type=str, default='IS_IP_SC',help="Problem type to train on (e.g., IS, WA, IP)")
    exp_group.add_argument("--test_num", type=int, default=100,
                         help="Number of test instances to process")
    exp_group.add_argument("--difficulty",default="hard", help="The difficulty of dataset")
    exp_group.add_argument("--gurobi_time",default="1000s", help="The time Gurobi takes to process the dataset")
    exp_group.add_argument("--real_world", action='store_true', default=False)
    
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument("--gnn_type", default="moe", choices=["gcn", "moe"],
                           help="GNN architecture type (default: %(default)s)")
    model_group.add_argument("--model_dir", default="./pretrain_models",
                           help="Directory containing pretrained models")
    
    gcn_group = parser.add_argument_group("GCN Settings")
    gcn_group.add_argument("--emb_size", type=int, default=64,
                       help="Embedding size for GNN (default: %(default)s)")
    gcn_group.add_argument("--constraint_nfeats", type=int, default=4, 
                       help="Number of features for constraint nodes (default: %(default)s)")
    gcn_group.add_argument("--edge_nfeats", type=int, default=1, 
                       help="Number of features for edge (default: %(default)s)")
    gcn_group.add_argument("--variable_nfeats", type=int, default=6, 
                       help="Number of features for variable nodes (default: %(default)s)")
    
    moe_group = parser.add_argument_group("MoE Settings")
    moe_group.add_argument("--num_shared_experts", type=int, default=1,
                       help="Number of shared experts in MoE (default: %(default)s)")
    moe_group.add_argument("--num_dedicate_experts", type=int, default=3,
                       help="Number of dedicated experts in MoE (default: %(default)s)")
    moe_group.add_argument("--top_k", type=int, default=1,
                       help="Top-K experts to select (default: %(default)s)")
    moe_group.add_argument("--eps_wasserstein", type=float, default=0.1,
                       help="Wasserstein ball radius for robust training (default: %(default)s)")
    moe_group.add_argument("--lb_alpha", type=float, default=0.1,
                       help="Load balance loss alpha (default: %(default)s)")
    moe_group.add_argument("--gate_temperature", type=float, default=1.0,
                       help="Gate temperature (default: %(default)s)")
    moe_group.add_argument('--dro_perturb_type', type=str, default="gaussian", choices=["gaussian", "uniform"],
                       help="Type of perturbation for DRO (default: %(default)s)")
    moe_group.add_argument('--robust_ratio', type=float, default=0.8)
    moe_group.add_argument('--other_loss_ratio', type=float, default=0.2)
    moe_group.add_argument('--bias_lr', type=float, default=1e-3,
                       help="Bias learning rate for gate network (default: %(default)s)")
    moe_group.add_argument('--use_dro', default=True, action='store_true',
                       help="Whether to use DRO (default: %(default)s)")

    # RoME configuration
    moe_group.add_argument('--robust', default=True, action='store_true')
    moe_group.add_argument('--generalization_adjustment', default="0.0")
    moe_group.add_argument('--robust_step_size', default=0.001, type=float)
    moe_group.add_argument('--use_normalized_loss', default=True, action='store_true')
    moe_group.add_argument('--gamma', type=float, default=0.1)

    solver_group = parser.add_argument_group("Solver Settings")
    solver_group.add_argument("--solver", choices=SOLVER_CLASSES.keys(), default="gurobi",
                            help="MIP solver implementation (default: %(default)s)")
    solver_group.add_argument("--max_time", type=int, default=1000,
                            help="Maximum solving time in seconds")
    solver_group.add_argument("--threads", type=int, default=1,
                            help="Number of threads for solving")
    solver_group.add_argument("--fix_strategy", choices=["pas", "soft_confidence"], default="pas",
                            help="Variable fixing strategy (default: %(default)s)")
 
    sys_group = parser.add_argument_group("System Settings")
    sys_group.add_argument("--instance_dir", default="./instance/test",
                         help="Path to test instances directory")
    sys_group.add_argument("--log_dir", default="./test_logs/",
                         help="Path to test instances directory")
    sys_group.add_argument("--scores_dir", default="./rome_best_scores",
                         help="Path to store scores directory")
    sys_group.add_argument("--device", default="cuda:6",
                         help="Computation device (default: %(default)s)")
    sys_group.add_argument("--num_workers", type=int, default=1,
                         help="Number of parallel workers (default: %(default)s)")
    sys_group.add_argument('--load_scores', default=False, action='store_true')
    sys_group.add_argument("--time_flag", default="20251126_234927",
                         help="Computation device (default: %(default)s)")
    
    return parser.parse_args()

def worker_process(task_queue, args, scores_dir, log_dir, save_name):
    logger = configure_logging(log_dir=log_dir)
    
    while True:
        ins_name = task_queue.get()
        if ins_name is None:
            break
        
        try:
            ins_path = os.path.join(args.instance_dir, args.test_problem_type, args.difficulty, ins_name)
            log_path = os.path.join(log_dir, f"{ins_name}.log")
            if args.load_scores:
                with open(os.path.join(args.instance_dir, f'scores_{args.test_problem_type}', f"scores_{ins_name.rsplit('.', 1)[0]}.pkl"), "rb") as f:
                    scores = pickle.load(f)
            else:
                score_path = os.path.join(scores_dir, f"scores_{ins_name.rsplit('.', 1)[0]}.pkl")
                with open(score_path, "rb") as f:
                    scores = pickle.load(f)
            
            logger.info(f"Start solving {ins_name} with {args.solver}")
            solve_mps(ins_path, log_path, save_name, ins_name, scores, args.test_problem_type, args)
            
        except Exception as e:
            logger.error(f"Error processing {ins_name}: {str(e)}")

def main():
    setup_environment()
    args = parse_arguments()

    if args.gnn_type == "moe":
        save_name = f'{args.time_flag}_{args.gnn_type}_shared_{args.num_shared_experts}_dedicate_{args.num_dedicate_experts}'
    elif args.gnn_type == "gcn":
        save_name = f'{args.time_flag}_{args.gnn_type}'
    model_path = os.path.join(args.model_dir, args.method_type, args.training_problem_types, f"{save_name}_model_best.pth")
    policy = load_pretrained_model(args, model_path, args.device)

    log_dir = os.path.join(args.log_dir, args.method_type, args.solver, args.test_problem_type, args.difficulty, args.gurobi_time, 
                         f"{save_name}")
    os.makedirs(log_dir, exist_ok=True)

    scores_dir = os.path.join(args.scores_dir, args.method_type, args.solver, args.test_problem_type, args.difficulty, args.gurobi_time, 
                         f"{save_name}")
    os.makedirs(scores_dir, exist_ok=True)

    test_instances = sorted(os.listdir(os.path.join(args.instance_dir, args.test_problem_type, args.difficulty)))
    if not args.load_scores:
        for ins_name in test_instances:
            ins_path = os.path.join(args.instance_dir, args.test_problem_type, args.difficulty, ins_name)
            file_path = os.path.join(scores_dir, f"scores_{ins_name.rsplit('.', 1)[0]}.pkl")
            if os.path.exists(file_path):
                continue
            scores = process_single_instance(args.gnn_type, ins_path, policy, args.device)
            with open(file_path, 'wb') as f:
                pickle.dump(scores, f)

    task_queue = Queue()

    for ins_name in test_instances:
        task_queue.put(ins_name)
    
    num_workers = args.num_workers

    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for _ in range(num_workers):
        p = Process(
            target=worker_process,
            args=(task_queue, args, scores_dir, log_dir, save_name)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print("Testing completed successfully.")

if __name__ == "__main__":
    main()

    