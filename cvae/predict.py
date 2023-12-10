import argparse

from cvae import CVAE
from baseline import BaselineNet
import pandas as pd
import torch
from util import generate_table, visualize
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fire

import pyro

torch.set_default_dtype(torch.float32)

def main(args):
    pyro.set_rng_seed(args.seed)
    # seeds with good examples: 2

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')
    results = []
    columns = []

    input_size = 128 # height of fire images
    input_dim = 4*input_size**2
    cvae_model_path_100 = "cvae_net_z100.pth"
    cvae_model_path_1000 = "cvae_net_z1000.pth"
    baseline_model_path = "baseline_net.pth"

    cvae_net_100 = CVAE(input_dim, 100, 1000, 1000).to(device)
    cvae_net_100.load_state_dict(torch.load(cvae_model_path_100))

    cvae_net_1000 = CVAE(input_dim, 1000, 1000, 1000).to(device)
    cvae_net_1000.load_state_dict(torch.load(cvae_model_path_1000))

    baseline_net = BaselineNet(input_dim, 100, 100).to(device)
    baseline_net.load_state_dict(torch.load(baseline_model_path))

    data_path = "../data/cropped_frame_pairs_with_external.pt"

    # Visualize conditional predictions
    visualize(
        device=device,
        input_size=input_size,
        pre_trained_baseline=baseline_net,
        pre_trained_cvae=cvae_net_100,
        num_images=args.num_images,
        num_samples=args.num_samples,
        data_path=data_path,
        image_path="cvae_plot.png",
    )
    # Retrieve conditional log likelihood
    df = generate_table(
        device=device,
        pre_trained_baseline=baseline_net,
        pre_trained_cvae_100=cvae_net_100,
        pre_trained_cvae_1000=cvae_net_1000,
        num_particles=args.num_particles,
        col_name="fire",
        data_path=data_path,
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
        "-vi",
        "--num-images",
        default=3,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=3,
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
    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="seed for reproducibility",
    )
    args = parser.parse_args()

    main(args)