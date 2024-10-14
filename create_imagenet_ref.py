from functools import partial
import os
import argparse
import yaml
import types
import lpips

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_pigdm import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from data.ImageNetDataModule import ImageNetDataModule
from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import Deblurring, Inpainting, Denoising, Deblurring2D, Colorization, SuperResolution, SRConv
from util.inpaint.get_mask import MaskCreator
from torchmetrics.functional import peak_signal_noise_ratio


def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # torch.set_default_dtype(torch.float64)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--imagenet', action='store_true')
    args = parser.parse_args()
    seed_everything(1, workers=True)

    # logger
    logger = get_logger()

    task_config = load_yaml(args.task_config)

    # Prepare dataloader
    dm = ImageNetDataModule(load_object(task_config))

    dm.setup()
    test_loader = dm.train_dataloader()

    for i, data in enumerate(test_loader):
        logger.info(f"Saving image {i}")
        y, x, _, mean, std = data[0]

        # TODO: Save x...
        if i < 1000:
            plt.imsave(
                f'/storage/imagenet_val/{i:05}.png',
                clear_color(x[0].unsqueeze(0)))

        plt.imsave(
            f'/storage/imagenet_ref/{i:05}.png',
            clear_color(x[0].unsqueeze(0)))


if __name__ == '__main__':
    main()
