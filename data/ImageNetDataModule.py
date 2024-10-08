from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from typing import Optional
from util.inpaint.get_mask import MaskCreator
from util.img_utils import center_crop

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

        # TODO: Process imagenet gt...
        print(gt_im.shape)
        exit()

        width = gt_im.shape[2]
        height = gt_im.shape[1]

        new_width = 256
        new_height = 256

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        center_cropped_img = img[..., top:bottom, left:right]


        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt * mask

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
        full_data = datasets.ImageFolder('/storage/ImageNet/imagenet_1k_val', transform=transform)
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
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
