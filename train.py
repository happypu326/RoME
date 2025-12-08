import argparse
import os
import numpy as np
import torch
import torch_geometric
import random
import time
from datetime import datetime

from model.gcn import GNNPolicy
from model.moe import MoEPolicy
from model.loss import LossComputer
from dataset.graph_dataset import GraphDataset
from utils.utils import TASKS, get_group_stats

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Predefined task-specific parameters
PROBLEM_CLASS = {"IS": 0, "IP": 1, "SC": 2, "MVC": 3, "CA": 4, "WA": 5}
GROUP_CLASS = {0: "IS", 1: "IP", 2: "SC", 3: "MVC", 4: "CA", 5: "WA"}
TASK_BATCH_SIZE = 1
ENERGY_WEIGHT_NORM = {"IP": 10, "WA": 100, "IS": -100, 'CA': -10000, 'SC': 100, "MVC": 100}


def train(method_type, gnn_type, predict, data_loader, optimizer=None, loss_computer=None,other_loss_ratio=0.1, device='cpu'):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)
            group = batch.group
            # Get target solutions in list format
            solInd = batch.nsols
            target_sols = []
            target_vals = []
            solEndInd = 0
            valEndInd = 0

            batch_indices = []
            for i in range(solInd.shape[0]):
                nvar = len(batch.varInds[i][0][0])
                batch_indices.extend([i] * nvar)
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
            # Predict the binary distribution, BD
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                torch.tensor(batch_indices, device=device),
                is_training=True
            )
            if gnn_type == 'moe':
                BD, other_loss = BD
                
            BD = BD.sigmoid()

            # Compute loss
            loss = 0
            loss_list = []

            # Calculate weights
            index_arrow = 0
            for ind,(sols,vals) in enumerate(zip(target_sols,target_vals)):
                group_ind = group[ind].cpu().item()
                problem_ind = GROUP_CLASS[group_ind]
                weight_norm = ENERGY_WEIGHT_NORM.get(problem_ind, 1)
                
                # Compute weight
                n_vals = vals
                exp_weight = torch.exp(-n_vals/weight_norm)
                weight = exp_weight/exp_weight.sum()

                # Get a binary mask
                varInds = batch.varInds[ind]
                varname_map=varInds[0][0]
                b_vars=varInds[1][0].long()

                # Get binary variables
                sols = sols[:,varname_map][:,b_vars]

                # Cross-entropy
                n_var = batch.ntvars[ind]
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var
                
                pos_loss = -(pre_sols+ 1e-8).log()[None,:]*(sols==1).float()
                neg_loss = -(1-pre_sols + 1e-8).log()[None,:]*(sols==0).float()
                sum_loss = pos_loss + neg_loss

                sample_loss = sum_loss*weight[:,None]
                total_sample_loss = sample_loss.sum()

                loss_list.append(total_sample_loss)
            
            batch_loss = torch.stack(loss_list)
            if optimizer is not None:
                if method_type == 'RoME':
                    if gnn_type == 'moe':
                        loss = loss_computer.loss(batch_loss, group) + other_loss_ratio * other_loss
                    else:
                        loss = loss_computer.loss(batch_loss, group)
                else:
                    if gnn_type == 'moe':
                        loss = batch_loss.sum() + other_loss_ratio * other_loss
                    else:
                        loss = batch_loss.sum()
            else:
                loss = batch_loss.sum()
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss

def get_parser():
    """Create and return the argument parser with all configuration parameters"""
    parser = argparse.ArgumentParser(description="train for predict and search.")

    parser.add_argument("--method_type", default='RoME', choices=['PS', 'RoME'],
                       help="Training method type (default: %(default)s)")
    parser.add_argument("--problem_type", type=str, nargs='+', choices=TASKS, default=['IS', 'IP', 'SC'],
                       help="Problem type to train on (e.g., IS, WA, IP)") 
    parser.add_argument("--gnn_type", default='moe', choices=['gcn', 'moe'],
                       help="Type of GNN architecture (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate for optimizer (default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=150,
                       help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loader workers (default: %(default)s)")
    parser.add_argument("--data_dir", default="./dataset",
                       help="Base directory for dataset (default: %(default)s)")
    parser.add_argument("--model_save_dir", default="./pretrain_models",
                       help="Directory to save models (default: %(default)s)")
    parser.add_argument("--log_save_dir", default="./train_logs",
                       help="Directory to save logs (default: %(default)s)")
    parser.add_argument("--device",default="cuda:0", help="cuda device")

    # gcn configuration
    parser.add_argument("--emb_size", type=int, default=64,
                       help="Embedding size for GNN (default: %(default)s)")
    parser.add_argument("--constraint_nfeats", type=int, default=4, 
                       help="Number of features for constraint nodes (default: %(default)s)")
    parser.add_argument("--edge_nfeats", type=int, default=1, 
                       help="Number of features for edge (default: %(default)s)")
    parser.add_argument("--variable_nfeats", type=int, default=6, 
                       help="Number of features for variable nodes (default: %(default)s)")
    
    # MoE configuration
    parser.add_argument("--num_shared_experts", type=int, default=1,
                       help="Number of shared experts in MoE (default: %(default)s)")
    parser.add_argument("--num_dedicate_experts", type=int, default=5,
                       help="Number of dedicated experts in MoE (default: %(default)s)")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Top-K experts to select (default: %(default)s)")
    parser.add_argument("--eps_wasserstein", type=float, default=0.1,
                       help="Wasserstein ball radius for robust training (default: %(default)s)")
    parser.add_argument("--gate_temperature", type=float, default=1.0,
                       help="Gate temperature (default: %(default)s)")
    parser.add_argument('--dro_perturb_type', type=str, default="gaussian", choices=["gaussian", "uniform"],
                       help="Type of perturbation for DRO (default: %(default)s)")
    parser.add_argument('--other_loss_ratio', type=float, default=0.15)
    parser.add_argument('--bias_lr', type=float, default=1e-3,
                       help="Bias learning rate for gate network (default: %(default)s)")
    parser.add_argument('--use_dro', default=True, action='store_true',
                       help="Whether to use DRO (default: %(default)s)")
    parser.add_argument('--use_struct_tokens', default=False, action='store_true',
                       help="Whether to use structural tokens (default: %(default)s)")
    parser.add_argument('--num_struct_tokens', type=int, default=64,
                       help="Number of structural tokens (default: %(default)s)")
    parser.add_argument('--struct_token_dim', type=int, default=64,
                       help="Dimension of structural tokens (default: same as emb_size)")
    parser.add_argument('--hard_token_routing', default=False, action='store_true',
                       help="Whether to use hard token routing (default: %(default)s)")
    parser.add_argument('--token_topk', type=int, default=8,
                       help="Top-K tokens for hard routing (default: %(default)s)")
    
    # RoME congiguration
    parser.add_argument('--robust', default=True, action='store_true')
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--robust_step_size', default=0.001, type=float)
    parser.add_argument('--use_normalized_loss', default=True, action='store_true')
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # Model saving configuration
    parser.add_argument('--save_top_k', type=int, default=3,
                       help="Number of best models to keep (default: %(default)s)")
    
    # Early stopping configuration
    parser.add_argument('--patience', type=int, default=30,
                       help="Number of epochs to wait for improvement before stopping (default: %(default)s)")
    parser.add_argument('--min_delta', type=float, default=1e-6,
                       help="Minimum change in validation loss to qualify as improvement (default: %(default)s)")
    
    return parser

def main():
    # Parse arguments and setup configurations
    parser = get_parser()
    args = parser.parse_args()
    
    # Set device configuration
    device = args.device
    gnn_type = args.gnn_type
    method_type = args.method_type
    batch_size = TASK_BATCH_SIZE
    problem_type = ''
    if method_type == 'RoME':
        for pt in args.problem_type:
            problem_type += pt + '_'
        problem_type = problem_type[:-1]
    else:
        problem_type = args.problem_type[0]
    
    save_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{gnn_type}_shared_{args.num_shared_experts}_dedicate_{args.num_dedicate_experts}'

    # Create directories
    model_save_path = os.path.join(args.model_save_dir, method_type, problem_type)
    log_save_path = os.path.join(args.log_save_dir, method_type, problem_type)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    
    # Initialize logging
    log_file = open(f'{log_save_path}/{save_name}_train.log', 'wb')
    
    # Write parser arguments to log file
    log_file.write(b"=== Training Arguments ===\n")
    for arg, value in sorted(vars(args).items()):
        log_file.write(f"{arg}: {value}\n".encode())
    log_file.write(b"========================\n\n")
    log_file.flush()
    
    # Load dataset
    sample_files = []
    problem_types = args.problem_type if method_type == 'RoME' else [problem_type]
    for pt in problem_types:
        dir_bg = os.path.join(args.data_dir, pt, 'BG')
        dir_sol = os.path.join(args.data_dir, pt, 'solution')
        pt_group = PROBLEM_CLASS.get(pt, 10)
        sample_files.extend([
            (
                os.path.join(dir_bg, name),
                os.path.join(dir_sol, name.replace('bg', 'sol')),
                pt_group
            )
            for name in os.listdir(dir_bg)
        ])

    random.shuffle(sample_files)
    split_idx = int(0.8 * len(sample_files))
    train_files, valid_files = sample_files[:split_idx], sample_files[split_idx:]

    if method_type == 'RoME':
        # Initialize loss computer
        train_group = [x[2] for x in train_files] 
        valid_group = [x[2] for x in valid_files]
        train_group_stats = get_group_stats(train_group, device)
        valid_group_stats = get_group_stats(valid_group, device)

        adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
        assert len(adjustments) in (1, train_group_stats['n_groups'])
        if len(adjustments) == 1:
            adjustments = np.array(adjustments * train_group_stats['n_groups'])
        else:
            adjustments = np.array(adjustments)

        train_loss_computer = LossComputer(
            is_robust=args.robust,
            group_stats=train_group_stats,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            device=device)
    
    # Create data loaders
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    if gnn_type == 'gcn':
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
                          dro_perturb_type=args.dro_perturb_type,
                          use_struct_tokens=args.use_struct_tokens,
                          num_struct_tokens=args.num_struct_tokens,
                          struct_token_dim=args.struct_token_dim,
                          hard_token_routing=args.hard_token_routing,
                          token_topk=args.token_topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize model stack for top K best models
    best_models = []  # List of tuples: (valid_loss, epoch, model_path)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        if method_type == 'RoME':
            val_loss_computer = LossComputer(
                is_robust=args.robust,
                group_stats=valid_group_stats,
                step_size=args.robust_step_size,
                device=device)
            train_loss = train(method_type, gnn_type, model, train_loader, optimizer, loss_computer=train_loss_computer,other_loss_ratio=args.other_loss_ratio, device=device)
            valid_loss = train(method_type, gnn_type, model, valid_loader, None, loss_computer=val_loss_computer,other_loss_ratio=args.other_loss_ratio, device=device)
        else:
            train_loss = train(method_type, gnn_type, model, train_loader, optimizer, device=device)
            valid_loss = train(method_type, gnn_type, model, valid_loader, None, device=device)
        
        # Update top K best models stack
        model_path = os.path.join(model_save_path, f'{save_name}_epoch{epoch}_val{valid_loss:.6f}.pth')
        torch.save(model.state_dict(), model_path)
        
        # Add to best models list and maintain top K
        best_models.append((valid_loss, epoch, model_path))
        best_models.sort(key=lambda x: x[0])  # Sort by validation loss (ascending)
        
        # Keep only top K models
        if len(best_models) > args.save_top_k:
            # Remove the worst model (highest loss)
            worst_loss, worst_epoch, worst_path = best_models.pop()
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass
        
        # Update best model if needed and check early stopping
        if valid_loss < best_val_loss - args.min_delta:
            best_val_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, f'{save_name}_model_best.pth'))
            log_entry_best = f'@epoch{epoch}   New best model saved with validation loss: {valid_loss:.6f}\n'
            log_file.write(log_entry_best.encode())
        else:
            patience_counter += 1
            log_entry_patience = f'@epoch{epoch}   No improvement. Patience counter: {patience_counter}/{args.patience}\n'
            log_file.write(log_entry_patience.encode())
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(model_save_path, f'{save_name}_model_last.pth'))
        
        # Log progress
        log_entry =  f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{valid_loss}    TIME:{time.time() - start_time}\n'
        log_file.write(log_entry.encode())
        log_file.flush()
        
        # Check early stopping condition
        if patience_counter >= args.patience:
            early_stop_msg = f'\nEarly stopping triggered at epoch {epoch}. No improvement for {args.patience} consecutive epochs.\n'
            log_file.write(early_stop_msg.encode())
            log_file.flush()
            print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.6f}")
            break
    
    log_file.close()
    if patience_counter < args.patience:
        print("Training completed successfully.")
    else:
        print(f"Training stopped early at epoch {epoch} due to no improvement.")

if __name__ == '__main__':
    main()