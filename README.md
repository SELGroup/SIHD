# Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning

## Installation
```
conda env create -f environment.yml
conda activate hier_diffusion
pip install -e .
```

## Model Training

- The hierarchical diffusers can be trained in parrallel as follows:
```
# Train the high-layer diffusers
python scripts/train.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2 --node_height 2
python scripts/train.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2 --node_height 1
# Train the low-layer diffuser
python scripts/train.py --config config.locomotion_ll --dataset walker2d-medium-replay-v2 --node_height 0
```

- Train the value predictor:
```
# Train the value predictor for the high-layers diffusers
python scripts/train_values.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2 --node_height 2
python scripts/train_values.py --config config.locomotion_hl --dataset walker2d-medium-replay-v2 --node_height 1
# Train the value predictor for the low-layer diffuser
python scripts/train_values.py --config config.locomotion_ll --dataset walker2d-medium-replay-v2 --node_height 0
```

## Model Evaluation
To evaluate the model, follow the command provided below.
```
python scripts/hd_plan_guided.py --dataset walker2d-medium-replay-v2 --maximal_height 3
```
