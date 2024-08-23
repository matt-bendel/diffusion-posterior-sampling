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
from guided_diffusion.gaussian_diffusion_ddrm import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
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
    diffusion_config["timestep_respacing"] = 'ddim20'
    diffusion_config["sampler"] = 'ddim'
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

    # SR DAMPING: 0.2
    # BLUR DAMPING: 0.1

    operators = ['inpainting']
    # operators = ['blur_gauss']
    # operators = ['sr_bicubic4']

    noise_levels = [0.05]

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    lpips_vals = []
    psnr_vals = []

    for l in range(len(operators)):
        measure_config['noise']['sigma'] = noise_levels[l]
        measure_config['operator']['name'] = operators[l]
        noiser = get_noise(**measure_config['noise'])

        base_im_count = 0
        for i, data in enumerate(test_loader):
            logger.info(f"Inference for image {i}")
            y, x, mask, mean, std = data[0]

            y = x + torch.rand_like(x) * measure_config['noise']['sigma']

            ref_img = x.to(device)

            # mask = mask.to(device)
            mask = mask.to(device)
            mask = torch.ones(mask.shape).to(device)
            mask[:, :, 64:192, 64:192] = 0

            measurement_cond_fn = None #partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            inpainting = False
            sr = False
            coloring = False
            blur_by = 1
            deg = measure_config['operator']['name']

            if measure_config['operator']['name'] == 'inpainting':
                deg = 'inp_box'
                missing_r = torch.nonzero(mask[0, 0].reshape(-1) == 0).long().reshape(-1) * 3
                missing_g = missing_r + 1
                missing_b = missing_g + 1
                missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
                H = Inpainting(3, 256, missing, mask, device)
                inpainting = True
            elif measure_config['operator']['name'][:10] == 'sr_bicubic':
                sr = True
                factor = int(measure_config['operator']['name'][10:])
                blur_by = factor
                def bicubic_kernel(x, a=-0.5):
                    if abs(x) <= 1:
                        return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                    elif 1 < abs(x) and abs(x) < 2:
                        return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                    else:
                        return 0

                k = np.zeros((factor * 4))
                for q in range(factor * 4):
                    x = (1 / factor) * (q - np.floor(factor * 4 / 2) + 0.5)
                    k[q] = bicubic_kernel(x)
                k = k / np.sum(k)
                kernel = torch.from_numpy(k).float().to(device)
                H = SRConv(kernel / kernel.sum(), 3, 256, device, stride=factor)
            elif measure_config['operator']['name'] == 'blur_uni':
                H = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, 256, device)
            elif measure_config['operator']['name'] == 'blur_gauss':
                sigma = 3.0
                pdf = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)
                kernel = pdf(torch.arange(61) - 30).to(device)
                kernel = kernel / kernel.sum()
                H = Deblurring(kernel, 3, 256, device)
            elif measure_config['operator']['name'] == 'blur_aniso':
                sigma = 20
                pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
                kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                    device)
                sigma = 1
                pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
                kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                    device)
                H = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 3,
                                       256, device)
            elif measure_config['operator']['name'] == 'color':
                coloring = True
                H = Colorization(256, device)
            elif measure_config['operator']['name'][:2] == 'sr':
                sr = True
                blur_by = int(measure_config['operator']['name'][2:])
                H = SuperResolution(3, 256, blur_by, device)
            else:
                H = Denoising(3, 256, device)

            # y_n = operator.forward(ref_img, mask=mask)
            y_n = H.H(ref_img)
            y_n = noiser(y_n)

            for k in range(1):
                # Sampling
                sample = None
                with torch.no_grad():
                    x_start = torch.randn(ref_img.shape, device=device)
                    sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, mask=mask,
                                       noise_sig=measure_config['noise']['sigma'], meas_type=measure_config['operator']['name'], truth=ref_img).float()

                lpips_vals.append(loss_fn_vgg(sample, ref_img).mean().detach().cpu().numpy())
                psnr_vals.append(peak_signal_noise_ratio(sample, ref_img).mean().detach().cpu().numpy())

                y = H.H(ref_img)
                if inpainting or coloring:
                    y = H.Ht(y).view(ref_img.shape[0], ref_img.shape[1], ref_img.shape[2], ref_img.shape[3])
                else:
                    y = y.view(ref_img.shape[0], ref_img.shape[1], ref_img.shape[2] if not sr else ref_img.shape[2] // blur_by, ref_img.shape[3] if not sr else ref_img.shape[2] // blur_by)

                for j in range(sample.shape[0]):
                    plt.imsave(f'/storage/matt_models/ddrm/ffhq/{deg}_noisy/image_{i * y.shape[0] + j}.png',
                               clear_color(sample[j].unsqueeze(0)))

        print(f'Avg. LPIPS: {np.mean(lpips_vals)} +/- {np.std(lpips_vals) / len(lpips_vals)}')
        print(f'Avg. PSNR: {np.mean(psnr_vals)} +/- {np.std(psnr_vals) / len(psnr_vals)}')


if __name__ == '__main__':
    main()
