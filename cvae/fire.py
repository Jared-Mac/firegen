import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, functional

from pyro.contrib.examples.util import MNIST


class firedata(Dataset):
    def __init__(self, data_dict):
        self.input_data = data_dict["input"]
        self.output_data = data_dict["output"]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        # Assuming your data is in the form of lists or arrays
        input_sample = torch.tensor(self.input_data[index], dtype=torch.float32)
        output_sample = torch.tensor(self.output_data[index], dtype=torch.float32)
        
        return {"input": input_sample, "output": output_sample}


class ToTensor:
    def __call__(self, sample):
        sample["original"] = functional.to_tensor(sample["original"])
        sample["digit"] = torch.as_tensor(
            np.asarray(sample["digit"]), dtype=torch.int64
        )
        return sample


class MaskImages:
    """Mask the right half of the input which contains the next step
    of the fire data with -1 and add the target output in the sample dict as the complementary of the input.
    """

    def __init__(self, mask_with=-1):
        self.mask_with = mask_with

    def __call__(self, images):
        tensor = images.squeeze()
        out = tensor.detach().clone()
        N, h, w = tensor.shape

        # removes the left half of the image from the target output
        out[:, :, : w // 2] = self.mask_with

        # now, sets the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_with
        sample = {}
        sample["input"] = inp
        sample["output"] = out
        return sample


def get_data(path, batch_size, train_percent=0.8):
    frames = torch.load(path)
    mask = MaskImages()
    masked_frames = mask(frames)
    num_train_samples = int(train_percent*masked_frames["input"].shape[0])

    datasets, dataloaders, dataset_sizes = {}, {}, {}
    # split masked_frames dict into train and val 
    datasets["train"] = firedata({k: v[:num_train_samples] for k, v in masked_frames.items()})
    datasets["val"] = firedata({k: v[num_train_samples:] for k, v in masked_frames.items()})
    for mode in ["train", "val"]:
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )


    return datasets, dataloaders, dataset_sizes