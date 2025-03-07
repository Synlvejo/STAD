# STAD
A performance analysis tool for parallel programs that considers both spatial and temporal patterns within trace data.

## Requirements

```
Python              3.8.19
cuda                11.8
gensim              4.3.3
numpy               1.24.1
pandas              2.0.3
torch               2.1.0+cu118
torch_geometric     2.5.3
SimTrace
```

## Time Slice Generation

Install SimTrace

For usage, refer to the script [generate_graph](./scripts/generate_graph.sh)

The generated data structure is as follows:

```
MPI_profile
└── lammps_128_abnormal   Program Name
    └── 100ms_closed      Duration
        ├── graph         Node Information
        └── graph_edge    Edge Information
```

## Preprocess

```
python preprocess.py -p lammps_128_abnormal -d 100ms_closed -n 128
```

After execution, the vectorized node information will be saved as `node_feature.csv`.

```
MPI_profile
└── lammps_128_abnormal         Program Name
    └── 100ms_closed            Duration
        ├── graph               Node Information
        ├── graph_edge          Edge Information
        └── node_feature.csv    Vectorized Node Information
```

The input paths are hardcoded in `preprocess.py`, so remember to modify them.

For more details on the parameters, please refer to `preprocess.py`.

## Train

```
# If first execution 
mkdir checkpoints

python train.py -p lammps_128_abnormal -d 100ms_closed -n 128 -b 128
```

The trained model will be saved in [checkpoints](./checkpoints/).

The input paths are hardcoded in `train.py`, so remember to modify them.

For more details on the parameters, please refer to `train.py`.

## Predict

```
# If first execution 
mkdir results
mkdir results/scores
mkdir results/heatmaps

python predict.py -p lammps_128_normal -d 100ms_closed -n 128 -b 128
```

The computed anomaly scores will be saved in [./results/scores/](./results/scores/).

The original heatmap will be saved in [./results/heatmaps/](./results/heatmaps/).

The input paths are hardcoded in `predict.py`, so remember to modify them.

For more details on the parameters, please refer to `predict.py`.

## Analyze

```
# If first execution 
mkdir results/heatmaps_filter

python analyze.py -p lammps_128_normal -d 100ms_closed -n 128 -b 128
```

The anomaly scores after filtering based on a threshold will be saved in [./results/heatmaps_filter/](./results/heatmaps_filter/).

For more details on the parameters, please refer to `analyze.py`.
