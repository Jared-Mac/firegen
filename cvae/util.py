# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mnist import get_data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from pyro.infer import Predictive, Trace_ELBO

import fire


def imshow(inp, image_path=None):
    # plot images
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    space = np.ones((inp.shape[0], 50, inp.shape[2]))
    inp = np.concatenate([space, inp], axis=1)

    ax = plt.axes(frameon=False, xticks=[], yticks=[])
    ax.text(-300, 200, "Inputs:")
    ax.text(-300, 200 + 224 + 3, "Truth:")
    ax.text(-300, 200 + (224 + 3) * 3, "CVAE:")
    ax.imshow(inp)

    if image_path is not None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()

    plt.clf()


def visualize(
    device,
    pre_trained_cvae,
    num_images,
    num_samples,
    image_path=None,
):
    # Load sample random data
    datasets, _, dataset_sizes = fire.get_data("../data/frame_pairs.pt", batch_size=20)
    dataloader = DataLoader(datasets["val"], batch_size=num_images, shuffle=True)

    batch = next(iter(dataloader))
    inputs = batch["input"].to(device)
    outputs = batch["output"].to(device)
    originals = batch["original"].to(device)

    # Make predictions

    predictive = Predictive(
        pre_trained_cvae.model, guide=pre_trained_cvae.guide, num_samples=num_samples
    )
    cvae_preds = predictive(inputs)["y"].view(num_samples, num_images, 224, 448)

    # Predictions are only made in the pixels not masked. This completes
    # the input quadrant with the prediction for the missing quadrants, for
    # visualization purpose
    for i in range(cvae_preds.shape[0]):
        cvae_preds[i][outputs == -1] = inputs[outputs == -1]

    # adjust tensor sizes
    inputs = inputs.unsqueeze(1)
    inputs[inputs == -1] = 1
    cvae_preds = cvae_preds.view(-1, 224, 448).unsqueeze(1)

    # get only the right half of the predictions
    inputs = inputs[:, :, :, :224]
    originals = originals.unsqueeze(1)[:, :, :, 224:]
    cvae_preds = cvae_preds[:, :, :, 224:]

    # make grids
    inputs_tensor = make_grid(inputs, nrow=num_images, padding=0)
    originals_tensor = make_grid(originals, nrow=num_images, padding=0)
    separator_tensor = torch.ones((3, 5, originals_tensor.shape[-1])).to(device)
    cvae_tensor = make_grid(cvae_preds, nrow=num_images, padding=0)

    # add vertical and horizontal lines
    for tensor in [originals_tensor, cvae_tensor]:
        for i in range(num_images - 1):
            tensor[:, :, (i + 1) * 224] = 0.3

    for i in range(num_samples - 1):
        cvae_tensor[:, (i + 1) * 224, :] = 0.3

    # concatenate all tensors
    grid_tensor = torch.cat(
        [
            inputs_tensor,
            separator_tensor,
            originals_tensor,
            separator_tensor,
            cvae_tensor,
        ],
        dim=1,
    )
    # plot tensors
    imshow(grid_tensor, image_path=image_path)


def generate_table(
    device,
    pre_trained_cvae,
    num_particles,
    col_name,
):
    # Load sample random data
    datasets, dataloaders, dataset_sizes = fire.get_data("../data/frame_pairs.pt", batch_size=20)

    # Load sample data
    loss_fn = Trace_ELBO(num_particles=num_particles).differentiable_loss

    cvae_mc_cll = 0.0
    num_preds = 0

    df = pd.DataFrame(index=["CVAE (Monte Carlo)"], columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders["val"], desc="Generating predictions".ljust(20))
    for batch in bar:
        inputs = batch["input"].to(device)
        outputs = batch["output"].to(device)
        num_preds += 1

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll += loss_fn(
            pre_trained_cvae.model, pre_trained_cvae.guide, inputs, outputs
        ).detach().item() / inputs.size(0)

    df.iloc[0, 0] = cvae_mc_cll / num_preds
    return df
