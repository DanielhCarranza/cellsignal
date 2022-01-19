"""Experiment running framework."""

import argparse
import importlib

import numpy as np
import torch
import wandb

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

def main():
    """Run an experiment.
    
    Sample command:
    python training/run_experiment.py --max_epochs=3 --model=models.SimpleModel --dataset=datasets.RXRX1 --batch_size=32 --lr=0.01 --wandb_project=experiment_name --wandb_entity=username
    """
    pass

if __name__ == "__main__":
    main()
