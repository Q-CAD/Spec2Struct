import argparse

from atomdiff.models.prior import LitScoreNet
from atomdiff.datasets import *

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

def initialize_data_module(args):
    data_module = StructureDataModule(
        data_dir=args.data_dir,
        cutoff=args.cutoff,
        train_prior=args.train_prior,
        k=args.k,
        train_size=args.train_size,
        scale_y=args.scale_y,
        dup=args.dup,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        fractional_coors=args.fractional_coors
    )

    return data_module

def initialize_network(args):
    score_net = LitScoreNet(
        num_species=args.num_species, 
        num_convs=args.num_convs,
        dim=args.dim, 
        ema_decay=args.ema_decay, 
        learn_rate=args.learn_rate
    )

    return score_net

def main(args):
    data_module = initialize_data_module(args)
    score_net = initialize_network(args)

    trainer = L.Trainer(
        max_steps = args.max_steps,
        logger    = TensorBoardLogger(save_dir='./training_logs/', name=args.run_name),
        callbacks = [TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(score_net, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a prior network')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff radius for the structure')
    parser.add_argument('--train_prior', type=bool, default=True, help='Whether to train the prior')
    parser.add_argument('--k', type=float, default=0.8, help='k parameter for the structure')
    parser.add_argument('--train_size', type=float, default=0.9, help='Size of the train mask')
    parser.add_argument('--scale_y', type=float, default=1.0, help='Scaling factor for the target')
    parser.add_argument('--dup', type=int, default=1, help='Number of duplicates')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the data')
    parser.add_argument('--fractional_coors', type=bool, default=False, help='Whether to use fractional coordinates')
    parser.add_argument('--num_species', type=int, default=100, help='Number of species')
    parser.add_argument('--num_convs', type=int, default=5, help='Number of convolutions')
    parser.add_argument('--dim', type=int, default=200, help='Dimension of the network')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='Exponential moving average decay')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=100_000, help='Maximum number of steps')
    parser.add_argument('--run_name', type=str, default="default_run", help='Name of the run')

    args = parser.parse_args()
    main(args)
