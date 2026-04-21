# RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains

This repository contains the codebase for the **extended journal version** of **RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains** (NeurIPS 2025) .

Paper (conference): https://openreview.net/forum?id=VmK7gYaKXl

The journal extension is still being organized; updates will follow.

## Our Other Work

**MILP‑X** is a unified framework that enables training and testing with just a few commands. It also inherits data generation code for various datasets, allowing users to generate data seamlessly. You can access it here: https://github.com/happypu326/MILP-X

## What’s new in the journal extension

- **Upgraded MoE architecture**: top-`k` routing with shared + task-specific experts; adaptive routing without additional load-balancing loss.
- **Perturbation within Wasserstein ball**: task-embedding perturbations are constrained in a Wasserstein ball around the original embedding.
- **Expanded synthetic datasets**: broader training coverage with more diverse MILP structures.

## Environment Setup

- **Python**: 3.8.13
- **Gurobi**: 9.5.2
- **SCIP**: 8.0.1
- **NetworkX**: 2.8.4

To build the environment, use the provided Conda environment file:

```bash
conda env create -f environment.yml
```

## Project Structure

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

### Artifacts (Datasets / Logs )

- **Conference-version dataset**: https://huggingface.co/datasets/tianle326/L2O-MILP
- **Journal-extension dataset**: https://huggingface.co/datasets/tianle326/MILP-X
- **Journal-extension testing logs**: https://huggingface.co/datasets/tianle326/MILP-X

### Datasets

We consider three groups of datasets:

#### Synthetic datasets (training)

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

#### Synthetic datasets (testing)

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

#### Real-world datasets (testing)

In addition to synthetic benchmarks, we evaluate on real-world MILP datasets:

- **MIRP**: Maritime Inventory Routing
- **MMCN**: Middle-Mile Consolidation Network Design
- **NNV**: Neural Network Verification
- **OTS**: Optimal Transmission Switching under High Wildfire Ignition Risk
- **SRPN**: Seismic-Resilient Pipe Network Planning

## Experimental Results

We report results from three evaluation settings:

1. **In-domain synthetic evaluation**: comparison with baseline methods on the synthetic tasks used during training.
2. **Out-of-domain synthetic evaluation**: comparison with Gurobi on unseen synthetic domains.
3. **Real-world evaluation**: testing on real-world MILP benchmarks.

**Improvement** are reported relative to the results obtained by running Gurobi with a 1000s time limit.

**BKS** denotes the result obtained by running **Gurobi with a 3600s time limit**.

### 1. In-domain synthetic evaluation (training domains)

<table style="border-collapse: collapse; margin: 0 auto; text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center; vertical-align:middle;">Method</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">BP<br>(BKS 12880.49)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">CA<br>(BKS 115461.28)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">CFLP<br>(BKS 25247.88)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">GISP<br>(BKS -4389.73)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">IP<br>(BKS 11.69)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">LB<br>(BKS 708.01)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">MIS<br>(BKS 2879.01)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">MVC<br>(BKS 214.93)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">SC<br>(BKS 395.84)</th>
    </tr>
    <tr>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Gurobi</td>
      <td style="text-align:center; vertical-align:middle;">12280.45</td><td style="text-align:center; vertical-align:middle;">162.94s</td>
      <td style="text-align:center; vertical-align:middle;">114782.40</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">25247.88</td><td style="text-align:center; vertical-align:middle;">148.41s</td>
      <td style="text-align:center; vertical-align:middle;">-4122.32</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">12.00</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">708.18</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">2875.00</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">214.93</td><td style="text-align:center; vertical-align:middle;">997.16s</td>
      <td style="text-align:center; vertical-align:middle;">395.88</td><td style="text-align:center; vertical-align:middle;">883.47s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">PS</td>
      <td style="text-align:center; vertical-align:middle;">12280.47</td><td style="text-align:center; vertical-align:middle;">139.86s</td>
      <td style="text-align:center; vertical-align:middle;">115016.28</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">25247.88</td><td style="text-align:center; vertical-align:middle;">116.35s</td>
      <td style="text-align:center; vertical-align:middle;">-4196.74</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">11.95</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">708.13</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">2878.20</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">214.93</td><td style="text-align:center; vertical-align:middle;">984.97s</td>
      <td style="text-align:center; vertical-align:middle;">395.87</td><td style="text-align:center; vertical-align:middle;">894.55s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Apollo</td>
      <td style="text-align:center; vertical-align:middle;">12280.47</td><td style="text-align:center; vertical-align:middle;">183.63s</td>
      <td style="text-align:center; vertical-align:middle;">114869.22</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">25247.88</td><td style="text-align:center; vertical-align:middle;">148.02s</td>
      <td style="text-align:center; vertical-align:middle;">-4182.65</td><td style="text-align:center; vertical-align:middle;">100.50s</td>
      <td style="text-align:center; vertical-align:middle;">11.91</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">708.17</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">2875.60</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">214.93</td><td style="text-align:center; vertical-align:middle;">100.08s</td>
      <td style="text-align:center; vertical-align:middle;">395.90</td><td style="text-align:center; vertical-align:middle;">922.39s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">CoCo</td>
      <td style="text-align:center; vertical-align:middle;">12280.44</td><td style="text-align:center; vertical-align:middle;">132.27s</td>
      <td style="text-align:center; vertical-align:middle;">114933.22</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">25247.88</td><td style="text-align:center; vertical-align:middle;">143.49s</td>
      <td style="text-align:center; vertical-align:middle;">-4249.92</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">11.96</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">708.13</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">2877.86</td><td style="text-align:center; vertical-align:middle;">1000.00s</td>
      <td style="text-align:center; vertical-align:middle;">214.93</td><td style="text-align:center; vertical-align:middle;">991.95s</td>
      <td style="text-align:center; vertical-align:middle;">395.85</td><td style="text-align:center; vertical-align:middle;">851.77s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;"><b>RoME</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>12280.47</b></td><td style="text-align:center; vertical-align:middle;"><b>121.18s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>115115.18</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>25247.88</b></td><td style="text-align:center; vertical-align:middle;"><b>92.66s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>-4342.84</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>11.90</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>708.12</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>2878.43</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>214.93</b></td><td style="text-align:center; vertical-align:middle;"><b>973.69s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>395.84</b></td><td style="text-align:center; vertical-align:middle;"><b>863.62s</b></td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Improvement</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">50%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">49.01%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">1.60x</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">82.46%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">32.25%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">85.53%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">35.29%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">1.02x</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">1.02x</td>
    </tr>
  </tbody>
</table>

### 2. Out-of-domain synthetic evaluation (unseen domains)

<table style="border-collapse: collapse; margin: 0 auto; text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center; vertical-align:middle;">Method</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">GC<br>(BKS 6.01)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">JS<br>(BKS 565.54)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">MC<br>(BKS 32168.48)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">NF<br>(BKS 345748.24)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">PF<br>(BKS 39.22)</th>
    </tr>
    <tr>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Gurobi</td>
      <td style="text-align:center; vertical-align:middle;">6.01</td><td style="text-align:center; vertical-align:middle;">435.12s</td>
      <td style="text-align:center; vertical-align:middle;">566.08</td><td style="text-align:center; vertical-align:middle;">727.31s</td>
      <td style="text-align:center; vertical-align:middle;">31817.09</td><td style="text-align:center; vertical-align:middle;">1000s</td>
      <td style="text-align:center; vertical-align:middle;">346233.2</td><td style="text-align:center; vertical-align:middle;">991.05s</td>
      <td style="text-align:center; vertical-align:middle;">39.17</td><td style="text-align:center; vertical-align:middle;">1000s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;"><b>RoME</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>6.01</b></td><td style="text-align:center; vertical-align:middle;"><b>212.61s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>565.97</b></td><td style="text-align:center; vertical-align:middle;"><b>687.60s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>31842.25</b></td><td style="text-align:center; vertical-align:middle;"><b>1000s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>346202.05</b></td><td style="text-align:center; vertical-align:middle;"><b>989.41s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>39.18</b></td><td style="text-align:center; vertical-align:middle;"><b>1000s</b></td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Improvement</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">2.04×</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">20.37%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">7.16%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">6.42%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">20%</td>
    </tr>
  </tbody>
</table>

### 3. Real-world evaluation

<table style="border-collapse: collapse; margin: 0 auto; text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center; vertical-align:middle;">Method</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">MMCN<br>(BKS 157756.18)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">MIRP<br>(BKS 37679.28)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">NNV<br>(BKS -14.14)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">OTS<br>(BKS 3.54)</th>
      <th colspan="2" style="text-align:center; vertical-align:middle;">SRPN<br>(BKS 51398198.76)</th>
    </tr>
    <tr>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↑</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
      <th style="text-align:center; vertical-align:middle;">Obj↓</th><th style="text-align:center; vertical-align:middle;">Time↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Gurobi</td>
      <td style="text-align:center; vertical-align:middle;">157840.92</td><td style="text-align:center; vertical-align:middle;">1000s</td>
      <td style="text-align:center; vertical-align:middle;">39672.12</td><td style="text-align:center; vertical-align:middle;">783.25s</td>
      <td style="text-align:center; vertical-align:middle;">-14.14</td><td style="text-align:center; vertical-align:middle;">7.64s</td>
      <td style="text-align:center; vertical-align:middle;">3.54</td><td style="text-align:center; vertical-align:middle;">87.85s</td>
      <td style="text-align:center; vertical-align:middle;">51411108.55</td><td style="text-align:center; vertical-align:middle;">1000s</td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;"><b>RoME</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>157823.80</b></td><td style="text-align:center; vertical-align:middle;"><b>998.77s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>38977.55</b></td><td style="text-align:center; vertical-align:middle;"><b>801.31s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>-14.14</b></td><td style="text-align:center; vertical-align:middle;"><b>7.08s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>3.54</b></td><td style="text-align:center; vertical-align:middle;"><b>60.40s</b></td>
      <td style="text-align:center; vertical-align:middle;"><b>51402212.08</b></td><td style="text-align:center; vertical-align:middle;"><b>1000.00s</b></td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:middle;">Improvement</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">20.20%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">34.85%</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">1.07×</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">1.45×</td>
      <td colspan="2" style="text-align:center; vertical-align:middle;">68.91%</td>
    </tr>
  </tbody>
</table>

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

