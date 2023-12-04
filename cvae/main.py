# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import cvae
import pandas as pd
import torch
from util import generate_table, get_data, visualize

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fire

import pyro

#set default dtype
torch.set_default_dtype(torch.float32)
pyro.set_rng_seed(1)

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')
    results = []
    columns = []

    # Dataset
    datasets, dataloaders, dataset_sizes = fire.get_data("../data/frame_pairs.pt", batch_size=20)

    # Train CVAE
    cvae_net = cvae.train(
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        model_path="cvae_net.pth",
    )

    # Visualize conditional predictions
    visualize(
        device=device,
        pre_trained_cvae=cvae_net,
        num_images=args.num_images,
        num_samples=args.num_samples,
        image_path="cvae_plot.png",
    )

    # Retrieve conditional log likelihood
    df = generate_table(
        device=device,
        pre_trained_cvae=cvae_net,
        num_particles=args.num_particles,
        col_name="fire",
    )
    results.append(df)
    columns.append("fire")

    results = pd.concat(results, axis=1, ignore_index=True)
    results.columns = columns
    results.to_csv("results.csv")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.6")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=3, type=int, help="early stop patience"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-5, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "-vi",
        "--num-images",
        default=10,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=10,
        type=int,
        help="number of samples to visualize per image",
    )
    parser.add_argument(
        "-p",
        "--num-particles",
        default=10,
        type=int,
        help="n of particles to estimate logpθ(y|x,z) in ELBO",
    )
    args = parser.parse_args()

    main(args)
