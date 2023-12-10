# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from baseline import MaskedBCELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from pyro.infer import Predictive, Trace_ELBO

import fire

def imshow(inp, input_size, image_path=None):
    # plot images
    inp = inp.cpu().numpy().transpose((2, 1, 0))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(170, -8, "Input")
    ax.text(430, -8, "Truth")
    ax.text(550, -8, "Baseline")
    ax.text(785, -8, "CVAE Samples")
    ax.imshow(inp[:,:,0], cmap="viridis")

    if image_path is not None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    else:
        plt.show()

    plt.clf()


def visualize(
    device,
    input_size,
    pre_trained_baseline,
    pre_trained_cvae,
    num_images,
    num_samples,
    data_path,
    image_path=None,
):
    # Load sample random data
    datasets, _, dataset_sizes = fire.get_data(data_path, batch_size=20)
    dataloader = DataLoader(datasets["val"], batch_size=num_images, shuffle=True)

    batch = next(iter(dataloader))
    inputs = batch["input"].to(device)
    outputs = batch["output"].to(device)
    originals = batch["original"].to(device)

    # Make predictions
    with torch.no_grad():
        baseline_preds = pre_trained_baseline(inputs).view(outputs.shape)

    predictive = Predictive(
        pre_trained_cvae.model, guide=pre_trained_cvae.guide, num_samples=num_samples
    )
    cvae_preds = predictive(inputs)["y"].view(num_samples, num_images, input_size, 4*input_size)

    # Predictions are only made in the pixels not masked. This completes
    # the input quadrant with the prediction for the missing quadrants, for
    # visualization purpose
    for i in range(cvae_preds.shape[0]):
        cvae_preds[i][outputs == -1] = inputs[outputs == -1]

    # adjust tensor sizes
    inputs = inputs.unsqueeze(1)
    inputs[inputs == -1] = 1
    cvae_preds = cvae_preds.view(-1, input_size, 4*input_size).unsqueeze(1)

    # get only the right half of the predictions
    fire_inputs = inputs[:, :, :, :input_size]
    wind_speed = inputs[:, :, :, input_size:2*input_size]
    wind_direction = inputs[:, :, :, 2*input_size:3*input_size]
    originals = originals.unsqueeze(1)[:, :, :, 3*input_size:]
    cvae_preds = cvae_preds[:, :, :, 3*input_size:]
    baseline_preds = baseline_preds.unsqueeze(1)[:, :, :, 3*input_size:]

    # make grids
    inputs_tensor = make_grid(fire_inputs, nrow=num_images, padding=0)
    wind_speed_tensor = make_grid(wind_speed, nrow=num_images, padding=0)
    wind_direction_tensor = make_grid(wind_direction, nrow=num_images, padding=0)
    originals_tensor = make_grid(originals, nrow=num_images, padding=0)
    separator_tensor = torch.ones((3, 5, originals_tensor.shape[-1])).to(device)
    cvae_tensor = make_grid(cvae_preds, nrow=num_images, padding=0)
    baseline_tensor = make_grid(baseline_preds, nrow=num_images, padding=0)

    # add vertical and horizontal lines
    for tensor in [originals_tensor, cvae_tensor]:
        for i in range(num_images - 1):
            tensor[:, :, (i + 1) * input_size] = 0.3

    for i in range(num_samples - 1):
        cvae_tensor[:, (i + 1) * input_size, :] = 0.3

    # concatenate all tensors
    grid_tensor = torch.cat(
        [
            inputs_tensor,
            wind_speed_tensor,
            wind_direction_tensor,
            separator_tensor,
            originals_tensor,
            separator_tensor,
            baseline_tensor,
            separator_tensor,
            cvae_tensor,
        ],
        dim=1,
    )
    # plot tensors
    imshow(grid_tensor, input_size, image_path=image_path)


def generate_table(
    device,
    pre_trained_baseline,
    pre_trained_cvae_100,
    pre_trained_cvae_1000,
    num_particles,
    col_name,
    data_path,
):
    # Load sample random data
    datasets, dataloaders, dataset_sizes = fire.get_data(data_path, batch_size=20)

    # Load sample data
    criterion = MaskedBCELoss()
    loss_fn = Trace_ELBO(num_particles=num_particles).differentiable_loss

    baseline_cll = 0.0
    cvae_mc_cll_100 = 0.0
    cvae_mc_cll_1000 = 0.0
    num_preds = 0

    df = pd.DataFrame(index=["NN (baseline)", "CVAE (z=100)","CVAE (z=1000)"], columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders["val"], desc="Generating predictions".ljust(20))
    for batch in bar:
        inputs = batch["input"].to(device)
        outputs = batch["output"].to(device)
        num_preds += 1

        # Compute negative log likelihood for the baseline NN
        with torch.no_grad():
            preds = pre_trained_baseline(inputs)
        baseline_cll += criterion(preds, outputs).item() / inputs.size(0)

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll_100 += loss_fn(
            pre_trained_cvae_100.model, pre_trained_cvae_100.guide, inputs, outputs
        ).detach().item() / inputs.size(0)

        cvae_mc_cll_1000 += loss_fn(
            pre_trained_cvae_1000.model, pre_trained_cvae_1000.guide, inputs, outputs
        ).detach().item() / inputs.size(0)

    df.iloc[0, 0] = baseline_cll / num_preds
    df.iloc[1, 0] = cvae_mc_cll_100 / num_preds
    df.iloc[2, 0] = cvae_mc_cll_1000 / num_preds
    return df
