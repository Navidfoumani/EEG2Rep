import os
import numpy as np
import pandas as pd
import argparse
import logging
# from running import Rep_Learning, Supervised

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# --------------------------------------------------- I/O --------------------------------------------------------------
parser.add_argument('--data_dir', default='Dataset/Crowdsource', help='Data directory')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Parameters and Hyperparameter ----------------------------------------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy'}, default='loss', help='Metric used for best epoch')
# -------------------------------------------------- EEG2Rep ----------------------------------------------------------
parser.add_argument('--Training_mode', default='Supervised', choices={'Rep-Learning', 'Initialization', 'Supervised'})
parser.add_argument('--Input_Embedding', default=['C'], choices={'T', 'C', 'C-T'}, help="Input Embedding Architecture")
parser.add_argument('--Pos_Embedding', default=['Sin'], choices={'Sin', 'Emb'}, help="Position Embedding Architecture")

parser.add_argument('--Encoder', default=['T'], choices={'T', 'C', 'C-T'}, help="Context/Target Encoder Architecture")
parser.add_argument('--layers', type=int, default=4, help="Number of layers for the context/target encoders")

parser.add_argument('--pre_layers', type=int, default=1, help="Number of layers for the Predictor")
parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")
parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")

parser.add_argument('--patch_size', type=int, default=16, help='size')
parser.add_argument('--emb_size', type=int, default=64, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of feedforward network of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()

