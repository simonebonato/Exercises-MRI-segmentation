# imports 
from utils.MedicalDecathlonClass import MedicalDecathlonDataModule
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

print('Last run on', time.ctime())

# extra imports
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-task", help="Task to run for the training of the model", type=str, default='Task04_Hippocampus')
parser.add_argument("-google_id", help="Google drive id for the datas", type=str, default="1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C")
parser.add_argument("-batch_size", help="Batch size for the training", type=int, default=16)
parser.add_argument("-train_val_ratio", help="Ratio of the training set to the validation set", type=float, default=0.8)
parser.add_argument("-epochs", help="Number of epochs for the training", type=int, default=100)
parser.add_argument("-lr", help="Learning rate for the training", type=float, default=0.001)
parser.add_argument("-early_stopping", help="Early stopping for the training. -1 if you don't want to use Early Stopping, else choose an integer number that will represent the patience.", type=int, default=10)
parser.add_argument("-best_models_dir", help="Directory where the best models will be saved", type=str, default='best_models')

args = parser.parse_args()

# asserts 
assert args.batch_size > 0, 'Batch size must be greater than 0'
assert args.train_val_ratio > 0 and args.train_val_ratio < 1, 'Train val ratio must be between 0 and 1'
assert args.epochs > 0, 'Epochs must be greater than 0'
assert args.lr > 0, 'Learning rate must be greater than 0'
assert isinstance(args.early_stopping, int), 'Early stopping must be an integer'
assert (args.early_stopping >= -1) and (args.early_stopping != 0), 'Early stopping can be -1 or greater than 0'
