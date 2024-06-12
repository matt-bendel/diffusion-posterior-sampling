from functools import partial
import os
import argparse
import yaml
import types

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_reform import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import Deblurring, Inpainting, Denoising


def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    seed_everything(1, workers=True)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    # operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_config['params']['scale'] = 2.0
    # cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = None #cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader

    dm = FFHQDataModule(load_object(task_config))
    dm.setup()
    test_loader = dm.test_dataloader()

    print(measure_config['noise']['sigma'])

    # Do Inference
    print(len(test_loader))
    for k in range(1):
        base_im_count = 0
        for i, data in enumerate(test_loader):
            if i <= 17:
                continue

            logger.info(f"Inference for image {i}")
            y, x, mask, mean, std = data[0]

            y = x + torch.rand_like(x) * measure_config['noise']['sigma']

            if i == 0 or i == 1:
                y_np = (y[0] * std[0, :, None, None] + mean[0, :, None, None]).cpu().numpy()
                plt.imshow(np.transpose(y_np, (1, 2, 0)))
                plt.savefig(f'y_noise_test.png')

            ref_img = x.to(device)

            # mask = mask.to(device)
            mask = torch.ones(mask.shape).to(device)
            mask[:, :, 104:184, 80:160] = 0

            measurement_cond_fn = None #partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            missing_r = torch.nonzero(mask[0, 0].reshape(-1) == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

            if measure_config['operator']['name'] == 'inpainting':
                H = Inpainting(3, 256, missing, device)
            elif measure_config['operator']['name'] == 'blur_uni':
                H = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, 256, device)
            elif measure_config['operator']['name'] == 'blur_gauss':
                sigma = 10
                pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
                kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
                H = Deblurring(kernel / kernel.sum(), 3, 256, device)
            else:
                H = Denoising(3, 256, device)

            # y_n = operator.forward(ref_img, mask=mask)
            y_n = H.H(ref_img)
            # y_n = ref_img
            y_n = noiser(y_n)

            for k in range(16):
                # Sampling
                with torch.no_grad():
                    x_start = torch.randn(ref_img.shape, device=device)
                    sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path, mask=mask,
                                       noise_sig=measure_config['noise']['sigma'], meas_type=measure_config['operator']['name'])

                # sample = ref_img * mask + (1 - mask) * sample
                # plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
                # plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))

                y = H.Ht(H.H(ref_img)).view(ref_img.shape[0], ref_img.shape[1], ref_img.shape[2], ref_img.shape[3])
                # y_n = ref_img
                y = noiser(y)
                for j in range(sample.shape[0]):
                    if j == 0:
                        plt.imsave(f'{measure_config["operator"]["name"]}/test_recon_{k}.png', clear_color(sample[j].unsqueeze(0)))
                        plt.imsave(f'{measure_config["operator"]["name"]}/test_y_{k}.png', clear_color(y[j].unsqueeze(0)))
                        plt.imsave(f'{measure_config["operator"]["name"]}/test_x_{k}.png', clear_color(ref_img[j].unsqueeze(0)))

                        if k > 14:
                            exit()

            base_im_count += sample.shape[0]


if __name__ == '__main__':
    main()
