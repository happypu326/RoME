# RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains

This repository contains the codebase for the **extended journal version** of **RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains** (NeurIPS 2025) .

Paper (conference): https://openreview.net/forum?id=VmK7gYaKXl

The journal extension is still being organized; updates will follow.

# What’s new in the journal extension

- **Upgraded MoE architecture**: top-`k` routing with shared + task-specific experts; adaptive routing without additional load-balancing loss.
- **Perturbation within Wasserstein ball**: task-embedding perturbations are constrained in a Wasserstein ball around the original embedding.
- **Expanded synthetic datasets**: broader training coverage with more diverse MILP structures.

# Environment Setup

- **Python**: 3.8.13
- **Gurobi**: 9.5.2
- **SCIP**: 8.0.1
- **NetworkX**: 2.8.4

To build the environment, use the provided Conda environment file:

```bash
conda env create -f environment.yml
```

# Project Structure

The workspace is organized as follows:

```
RoME/
├── dataset/                # dataset directory
├── data/                   # processed data cache
├── test_logs/              # output directory and log files
├── train_logs/             # output directory and log files
├── pretrain_models/        # pretrained models directory
├── solver/                 # solvers directory, including gurobi and scip
├── model/                  # GNN modules
│   ├── gcn.py              # gcn module
│   ├── moe.py              # moe module
│   ├── loss.py             # dro loss module
├── scripts/                # shell scripts for batch processing
│   ├── process.sh          # preprocessing script
│   ├── train.sh            # training script
│   └── test.sh             # testing script
├── preprocess.py           # preprocess MILP instances
├── calculate_objective.py  # calculate the objectives
├── train.py                # training
├── test.py                 # testing
├── utils/                  # utils files
│   ├── utils.py            # utility functions
├── environment.yml         # conda environment file
└── README.md
```

## Artifacts (Datasets / Logs )

- **Conference-version dataset**: https://huggingface.co/datasets/tianle326/L2O-MILP
- **Journal-extension dataset**: https://huggingface.co/datasets/tianle326/RoME
- **Journal-extension testing logs**: https://huggingface.co/datasets/tianle326/RoME

## Datasets

We consider three groups of datasets:

### Synthetic datasets (training)

RoME is trained on the following synthetic datasets:

- **BP**: Bin Packing
- **CA**: Combinatorial Auction
- **CFLP**: Capacitated Facility Location
- **GISP**: Generalized Independent Set
- **IP**: Balanced Item Placement
- **LB**: Load Balancing
- **MIS**: Maximum Independent Set
- **MVC**: Minimum Vertex Cover
- **SC**: Set Cover

### Synthetic datasets (testing)

RoME is evaluated on the following synthetic datasets:

- **BP**: Bin Packing
- **CA**: Combinatorial Auction
- **CFLP**: Capacitated Facility Location
- **GC**: Graph Coloring
- **GISP**: Generalized Independent Set
- **IP**: Balanced Item Placement
- **JS**: Job-Shop Scheduling
- **LB**: Load Balancing
- **MC**: Max Cut
- **MIS**: Maximum Independent Set
- **MVC**: Minimum Vertex Cover
- **NF**: Multicommodity Network Flow
- **PF**: Protein Folding
- **SC**: Set Cover

### Real-world datasets (testing)

In addition to synthetic benchmarks, we evaluate on real-world MILP datasets:

- **MIRP**: Maritime Inventory Routing
- **MMCN**: Middle-Mile Consolidation Network Design
- **NNV**: Neural Network Verification
- **OTS**: Optimal Transmission Switching under High Wildfire Ignition Risk
- **SRPN**: Seismic-Resilient Pipe Network Planning

## Usage

### Shell Scripts

The `scripts/` directory contains batch processing scripts for convenience:

- **process.sh**: Preprocessing script with example configurations
- **train.sh**: Training script with full parameter settings
- **test.sh**: Testing script with example test configurations

You can modify these scripts according to your needs or follow the step-by-step instructions below.

### 1. Data generation

We generate synthetic instances using [Ecole](https://www.ecole.ai/), obtain instances from existing generators (e.g., the ML4CO 2021 competition [generator](https://github.com/ds4dm/ml4co-competition-hidden) and Distributional [MIPLIB](https://sites.google.com/usc.edu/distributional-miplib/home)), and also generate instances using our own framework **[MILP‑X](https://github.com/happypu326/MILP-X)**.

For each benchmarks, we generate 300 instances for training and 100 instances for testing. 

### 2. Preprocessing

To preprocess a dataset (e.g., `CA`), run:

```bash
python preprocess.py --mode train --max_time 3600 --workers 20 --task "CA" --difficulty "hard"
```

The corresponding bipartite graph (BG) and solution will be automatically generated in the dataset folder.

### 3. Train

As an example, we train the model using the following nine synthetic datasets: BP, CA, CFLP, GISP, IP, LB, MIS, MVC, SC. The training command is:

```bash
python train.py \
  --method_type RoME \
  --problem_type BP CA CFLP GISP IP LB MIS MVC SC \
  --gnn_type moe \
  --device cuda:2 \
  --num_dedicate_experts 16 \
  --num_shared_experts 2 \
  --top_k 8 \
  --eps_wasserstein 0.2 \
  --lr 0.0001 \
  --num_epochs 80 \
  --patience  20 \
  --min_delta 0.0001
```

We have uploaded the pretrained model (trained on the above nine datasets with the specified hyperparameters) into the `pretrain_models` folder.

### 4. Test

To evaluate the trained model, run: 

```bash
python test.py \
    --time_flag [time flag] \
    --method_type "RoME" \
    --test_problem_type "CA" \
    --difficulty "hard" \
    --training_problem_types "BP_CA_CFLP_GISP_IP_LB_MIS_MVC_SC" \
    --test_num 100 \
    --gnn_type "moe" \
    --solver "gurobi" \
    --device "cpu" \
    --emb_size 64 \
    --num_shared_experts 2 \
    --num_dedicate_experts 16 \
    --top_k 8 \
    --max_time 1000 \
    --num_workers 20 \
    --instance_dir "/data/RoME/dataset/test"
```

Note: To avoid the impact of physical machine specifications on experimental results, it is recommended that the number of processes configured during preprocessing and testing does not exceed the maximum number of threads supported by the machine during **preprocess and testing**. For parallel testing, we pre-cache the scores of all instances in the `best_scores` folder. You can decide whether to enable this caching mechanism based on your specific use case.

## Citation

If you find RoME useful or relevant to your research, please consider citing our paper.

```
@inproceedings{pu2025rome,
  title={RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains},
  author={Pu, Tianle and Geng, Zijie and Liu, Haoyang and Liu, Shixuan and Wang, Jie and Zeng, Li and Chen, Chao and Fan, Changjun},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=VmK7gYaKXl}
}
```

# Our Related Work

**MILP‑X** is a unified framework that enables training and testing with just a few commands. It also inherits data generation code for various datasets, allowing users to generate data seamlessly. You can access it here: https://github.com/happypu326/MILP-X