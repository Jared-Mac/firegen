import argparse

from cvae import CVAE
import pandas as pd
import torch
from util import generate_table, predict
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
    model_path = "cvae_net.pth"

    cvae_net = CVAE(4*input_size**2, 1000, 3000, 3000)
    cvae_net.to(device)
    cvae_net.load_state_dict(torch.load(model_path))

    data_path = "../data/frame_pairs_with_external.pt"

    # Visualize conditional predictions
    predict(
        device=device,
        input_size=input_size,
        pre_trained_cvae=cvae_net,
        num_images=args.num_images,
        num_samples=args.num_samples,
        data_path=data_path,
        image_path="cvae_plot.png",
    )

    # Retrieve conditional log likelihood
    df = generate_table(
        device=device,
        pre_trained_cvae=cvae_net,
        num_particles=args.num_particles,
        col_name="fire",
        data_path=data_path
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
        default=10,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=5,
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