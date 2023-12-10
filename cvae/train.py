import argparse

import cvae
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fire
import baseline
import pyro
import matplotlib.pyplot as plt

#set default dtype
torch.set_default_dtype(torch.float32)
pyro.set_rng_seed(0)

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')
    results = []
    columns = []

    data_path = "../data/cropped_frame_pairs_with_external.pt"
    input_size = 128 # height of fire images
    z_dim = 100
    batch_size = 40

    # Dataset
    datasets, dataloaders, dataset_sizes = fire.get_data(data_path, batch_size=batch_size)

    # Train baseline
    baseline_net, baseline_loss = baseline.train(
        device=device,
        input_size=input_size,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        model_path="baseline_net.pth",
    )

    # Train CVAE
    cvae_net, cvae_loss = cvae.train(
        device=device,
        input_size=input_size,
        z_dim=z_dim,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        model_path="cvae_net_z{}.pth".format(z_dim),
    )

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.6")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=51, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=5, type=int, help="early stop patience"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-4, type=float, help="learning rate"
    )

    args = parser.parse_args()

    main(args)
