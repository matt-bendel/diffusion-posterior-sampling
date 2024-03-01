from functools import partial
import os
import argparse
import yaml
import types

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from pytorch_lightning import seed_everything


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
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
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
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()

    # Do Inference
    base_im_count = 0
    for i, data in enumerate(test_loader):
        logger.info(f"Inference for image {i}")
        y, x, mask, mean, std = data[0]

        ref_img = x.to(device)

        mask = mask.to(device)
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
        print(sample.shape)

        # plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        # plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        for j in range(sample.shape[0]):
            torch.save(sample[j].detach().cpu(), f'/storage/matt_models/inpainting/dps/test/image_{base_im_count+j}_sample_{0}.pt')
            torch.save(mask[j].detach().cpu(), f'/storage/matt_models/inpainting/dps/test/image_{base_im_count+j}_mask.pt')

            if i == 0 and j == 0:
                plt.imsave(f'/storage/matt_models/inpainting/dps/test_{i}.png', clear_color(sample[l].unsqueeze(0)))

        base_im_count += sample.shape[0]

if __name__ == '__main__':
    main()
