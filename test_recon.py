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
from guided_diffusion.gaussian_diffusion_vamp import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import Deblurring, Inpainting, Denoising, Deblurring2D, Colorization, SuperResolution, SRConv
from util.inpaint.get_mask import MaskCreator
from guided_diffusion.vamp_models_subspace import VAMP


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

    operators = ['sr_bicubic4', 'sr_bicubic8', 'blur_uni', 'blur_gauss', 'blur_aniso', 'color', 'sr4', 'sr8', 'denoising']
    noise_levels = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1]

    # operators = ['sr_bicubic8', 'color', 'inpainting']
    # noise_levels = [0.01, 0.01, 0.01, 0.01]

    operators = ['blur_uni']
    noise_levels = [0.01]

    for l in range(len(operators)):
        measure_config['noise']['sigma'] = noise_levels[l]
        measure_config['operator']['name'] = operators[l]
        noiser = get_noise(**measure_config['noise'])

        base_im_count = 0
        output_var_curves = []
        for i, data in enumerate(test_loader):
            print(i)
            if i <= 16: #i <= 3:
                continue

            logger.info(f"Inference for image {i}")
            y, x, mask, mean, std = data[0]

            y = x + torch.rand_like(x) * measure_config['noise']['sigma']

            if i == 0 or i == 1:
                y_np = (y[0] * std[0, :, None, None] + mean[0, :, None, None]).cpu().numpy()
                plt.imshow(np.transpose(y_np, (1, 2, 0)))
                plt.savefig(f'y_noise_test.png')

            ref_img = x.to(device)

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

            if measure_config['operator']['name'] == 'inpainting':
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
                for i in range(factor * 4):
                    x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                    k[i] = bicubic_kernel(x)
                k = k / np.sum(k)
                kernel = torch.from_numpy(k).float().to(device)
                H = SRConv(kernel / kernel.sum(), 3, 256, device, stride=factor)
            elif measure_config['operator']['name'] == 'blur_uni':
                H = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, 256, device)
            elif measure_config['operator']['name'] == 'blur_gauss':
                sigma = 10
                pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
                kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
                H = Deblurring(kernel / kernel.sum(), 3, 256, device)
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
            # y_n = ref_img
            y_n = noiser(y_n)

            for k in range(1):
                # Sampling
                with torch.no_grad():
                    x_start = ref_img
                    vamp_model = VAMP(model, sampler.betas_model, sampler.alphas_cumprod_model, 1, 1, x_start, H,
                                      inpainting=inpainting)

                    # t_vals = np.arange(1000)
                    # etas = []
                    # mse = []
                    # input_var = []
                    # for t in t_vals:
                    #     x_t = sampler.q_sample(x_start, t) / torch.sqrt(
                    #         torch.tensor(vamp_model.alphas_cumprod).to(x_start.device)[t])
                    #     noise_var = (1 - torch.tensor(vamp_model.alphas_cumprod).to(x_t.device)) / torch.tensor(
                    #         vamp_model.alphas_cumprod).to(x_t.device)
                    #     noise_var = noise_var[t].unsqueeze(0).repeat(x_t.shape[0], 1).float()
                    #     mu, true_noise_var, used_t = vamp_model.uncond_denoiser_function(x_t.float(), noise_var, False, False)
                    #     eta_2 = 1 / (vamp_model.scale_factor[used_t[0]] * true_noise_var.sqrt().repeat(x_t.shape[0],
                    #                                                                              vamp_model.Q)).float()[0,0].cpu().numpy()
                    #     etas.append(1/eta_2)
                    #
                    #     # plt.imsave(f'denoise_in_{t}.png', clear_color(x_t))
                    #     # plt.imsave(f'denoise_out_{t}.png', clear_color(mu))
                    #
                    #     # eta = vamp_model.denoiser_tr_approx(x_t, torch.tensor([1/noise_var[0, 0]]).to(mu.device).unsqueeze(0).repeat(x_t.shape[0], 1), mu, noise_var, False)
                    #     # etas.append(eta[0, 0].cpu().numpy())
                    #     mse.append(((vamp_model.mask[0, None, :, :, :] * (
                    #                 mu - x_start) ** 2).sum() / torch.count_nonzero(vamp_model.mask)).cpu().numpy())
                    #     input_var.append(noise_var[0, 0].cpu().numpy())
                    #
                    # plt.figure()
                    # plt.semilogy(t_vals, mse)
                    # plt.semilogy(t_vals, input_var)
                    # plt.semilogy(t_vals, np.sqrt(input_var))
                    # plt.semilogy(t_vals, etas)
                    # output_var_curves.append(mse)
                    # plt.xlabel('t')
                    # plt.legend(['MSE', 'Input variance', 'Sqrt Input variance', '1/eta_2 approx'])
                    # plt.savefig(f'vamp_debug/eta_2_approx/eta_2_debug_{base_im_count}.png')
                    # plt.close()


                    y = H.H(ref_img)
                    y = noiser(y)


                    plt.imsave('gt.png', clear_color(ref_img))
                    # plt.imsave('measures.png', clear_color(y.view(ref_img.shape[0], ref_img.shape[1], ref_img.shape[2] // blur_by, ref_img.shape[2] // blur_by)))

                    t_vals = [0, 25, 50, 100, 250, 500, 750, 999]
                    # t_vals = [25, 50, 100, 250]
                    # damping_factos = [0.1, 0.2, 0.5, 0.75, 1]
                    t_vals = [999]
                    damping_factos = [1]
                    for damp in damping_factos:
                        vamp_model.damping_factor = damp
                        for t in t_vals:
                            mse1s = []
                            mse2s = []
                            mser1s = []
                            mser2s = []

                            x_t = sampler.q_sample(x_start, t)
                            _, eta1s, eta2s, gam1s, gam2s, mu1s, mu2s, r1s, r2s = vamp_model.run_vamp_reverse_test(x_t, y, torch.tensor([t]).to(x_t.device), measure_config['noise']['sigma'], measure_config["operator"]["name"], ref_img, True)

                            for out in mu1s:
                                mse1s.append(torch.nn.functional.mse_loss(ref_img, out).item())

                            for out in mu2s:
                                mse2s.append(torch.nn.functional.mse_loss(ref_img, out).item())

                            for out in r1s:
                                mser1s.append(torch.nn.functional.mse_loss(ref_img, out).item())

                            for out in r2s:
                                mser2s.append(torch.nn.functional.mse_loss(ref_img, out).item())

                            plt.figure()
                            plt.semilogy(np.arange(4), eta1s, color='red')
                            plt.semilogy(np.arange(4), eta2s, color='blue')
                            plt.semilogy(np.arange(4), gam1s, color='green')
                            plt.semilogy(np.arange(4), gam2s, color='orange')
                            # plt.semilogy(np.arange(100), mse1s, linestyle='dashed', color='red')
                            # plt.semilogy(np.arange(100), mse2s, linestyle='dashed', color='blue')
                            # plt.semilogy(np.arange(100), mser1s, linestyle='dashed', color='green')
                            # plt.semilogy(np.arange(100), mser2s, linestyle='dashed', color='orange')
                            plt.xlabel('VAMP Iteration')
                            plt.legend(['1/eta_1', '1/eta_2', '1/gam_1', '1/gam_2', 'MSE mu_1', 'MSE mu_2', 'MSE r_1', 'MSE r_2'])
                            plt.title(measure_config['operator']['name'])
                            plt.savefig(f'vamp_debug/{measure_config["operator"]["name"]}/trajectories_t={t}_damp={damp}.png')
                            plt.close()

            break
                    # sample, g1_min, g1_max, g2_min, g2_max, e1_min, e1_max, e2_min, e2_max, mse_1, mse_2 = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path, mask=mask,
                    #                    noise_sig=measure_config['noise']['sigma'], meas_type=measure_config['operator']['name'], truth=ref_img)
            # base_im_count += 1
            # if base_im_count == 100:
            #     plt.figure()
            #     mean_out_var = np.mean(np.array(output_var_curves), axis=0)
            #     plt.semilogy(t_vals, mean_out_var)
            #     plt.semilogy(t_vals, np.sqrt(input_var))
            #     scale_factor = mean_out_var / np.sqrt(input_var)
            #     with open('eta_2_scale.npy', 'wb') as f:
            #         np.save(f, scale_factor)
            #     plt.semilogy(t_vals, scale_factor * np.sqrt(input_var))
            #     plt.xlabel('t')
            #     plt.legend(['output variance', 'sqrt(input_variance)'])
            #     plt.savefig(f'eta_2_debug_avg.png')
            #     plt.close()
            #     exit()
            # else:
            #     continue

if __name__ == '__main__':
    main()
