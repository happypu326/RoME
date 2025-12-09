## RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains

This is the code of paper **RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains.** Tianle Pu, Zijie Geng, Haoyang Liu, Shixuan Liu, Jie Wang, Li Zeng, Chao Chen, Changjun Fan. NeurIPS 2025.

Paper: https://arxiv.org/abs/2511.02331

**Note:** Building upon the conference paper version, we extend our work into a new version submitted to a journal. We are writing the paper now and release the latest code of the extended journal version. The code is still being organized and refined, and further updates will follow. Feel free to reach out if you have any questions.

**Here are the main improvements compared to the conference paper:**

- **Upgraded MoE Architecture**: We revise the MoE architecture to use top-$k$ routing with both shared and task-specific experts, enabling the model to learn compositional relationships among different constraint patterns. Load balancing is achieved through a loss-free, adaptive routing mechanism.

- **Perturbation within Wasserstein Ball**: Task embedding perturbations are constrained to lie within a Wasserstein ball centered at the original embedding.

- **TokenMemory for Generalization**: Patterns learned from synthetic datasets with single constraints are encoded as dedicated tokens in a TokenMemory module. At inference time (e.g., on MIPLIB instances), the model composes these tokens to handle problems with complex, mixed constraint types, thereby improving generalization.

- **Expanded Synthetic Datasets**: We extend the training data to include 8–10 diverse synthetic datasets, each modeling distinct constraint structures.

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
│   ├── graph_dataset.py    # graph dataset for MILP
├── instance/               # instances directory
├── test_logs/              # output directory and log files
├── train_logs/             # output directory and log files
├── pretrain_models         # pretrained models directory
├── solver                  # solvers directory, including guorbi and scip
├── model                   # GNN modules
│   ├── gcn.py              # gcn module 
│   ├── moe.py              # moe module 
│   ├── loss.py             # dro loss module 
├── preprocess.py           # precess MILP instances
├── calculate_objective.py  # calculate the objectives
├── train.py                # training
├── test.py                 # testing
├── utils/                  # utils files
│   ├── utils.py            # utility functions
├── environment.yml         # conda environment file
└── README.md
```

## Usage

### 1. Data generation

We use [Ecole](https://www.ecole.ai/) library to generate Independent Set (IS), Combinatorial Auction (CA) and Set Cover (SC) instance, and obtain the Balanced Item Placement (IP) and Workload Appointment (WA) instances from the ML4CO 2021 competition [generator](https://github.com/ds4dm/ml4co-competition-hidden). 

For each benchmarks, we generate 300 instances for training and 100 instances for testing. We take SC for example, after generating the instances, place them in the `instance` directory following this structure: `instance/train/SC` and `instance/test/SC`.

Note: we also collect more synthetic datasets for training in the journal version, like Minimum Vertex Cover, Multiple Knapsack, Bin Packing, Capacitated Facility Location, Generalized Independent Set and so on.

### 2. Preprocessing

To preprocess a dataset (e.g., `SC`), run:

```bash
python preprocess.py --problem_type "SC" --max_time 3600 --workers 10
```

The corresponding bipartite graph (BG) and solution will be automatically generated in the dataset folder.

### 3. Train

We take the same settings in the conference paper for example. We train the model using IS, IP and SC, run:

```bash
python train.py \
  --method_type RoME \
  --problem_type IS IP SC \
  --gnn_type moe \
  --device cuda:0 \
  --num_dedicate_experts 5 \
  --num_shared_experts 1 \
  --top_k 2 \
  --lr 0.0001 \
  --num_epochs 150
```

We upload the pretrained model on the IS, IP and SC using the above parameter settings into the `pretrain_models` folder.

### 4. Test

To evaluate the trained model, run: 

```bash
python test.py \
    --time_flag [time flag] \
    --method_type "RoME" \
    --test_problem_type "CA" \
    --training_problem_types "IS_IP_SC" \
    --test_num 100 \
    --gnn_type "moe" \
    --solver "gurobi" \
    --device "cuda:0" \
    --emb_size 64 \
    --num_shared_experts 1 \
    --num_dedicate_experts 5 \
    --top_k 2 \
    --max_time 1000 \
    --num_workers 16
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
  url={}
}
```


