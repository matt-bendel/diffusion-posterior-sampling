from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from typing import Optional
from util.inpaint.get_mask import MaskCreator
from util.img_utils import center_crop
from PIL import Image

import pathlib
import cv2
import torch
import numpy as np


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.args = args
        self.mask_creator = MaskCreator()

    def __call__(self, gt_im):
        mask1 = self.mask_creator.stroke_mask(self.args.image_size, self.args.image_size, max_length=self.args.image_size//2)
        mask2 = self.mask_creator.rectangle_mask(self.args.image_size, self.args.image_size, self.args.image_size//4, self.args.image_size//2)

        mask = mask1+mask2
        mask = mask > 0
        mask = mask.astype(np.float)
        mask = torch.from_numpy(1 - mask).unsqueeze(0)

        # arr = np.ones((256, 256))
        # arr[256 // 4: 3 * 256 // 4, 256 // 4: 3 * 256 // 4] = 0
        # mask = torch.tensor(np.reshape(arr, (256, 256)), dtype=torch.float).repeat(3, 1, 1)
        pil_image = Image.fromarray(np.uint8(np.transpose(gt_im.numpy(), (1, 2, 0)) * 255))
        image_size = 256

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        gt_im = np.transpose(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size, :], (2, 0, 1))

        # TODO: between -1,1
        gt = gt_im / 127.5 - 1

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        # gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt

        return masked_im.float(), gt.float(), mask.float(), mean.float(), std.float()


class ImageNetDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.big_test = big_test

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        transform = transforms.Compose([transforms.ToTensor(), DataTransform(self.args)])

        # Split into 1k val set for lr tune
        full_data = datasets.ImageFolder('/storage/ImageNet', transform=transform)
        test_data = torch.utils.data.Subset(full_data, range(0, 1000))

        self.full_data, self.lr_tune_data, self.test_data = test_data, test_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.full_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.lr_tune_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
