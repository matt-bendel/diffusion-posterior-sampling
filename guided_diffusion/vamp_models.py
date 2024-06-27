import torch
import numpy as np
import matplotlib.pyplot as plt
from guided_diffusion.ddrm_svd import Deblurring

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

class VAMP:
    def __init__(self, model, betas, alphas_cumprod, max_iters, K, x_T, svd, inpainting=False):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.max_iters = max_iters
        self.K = 1
        self.delta = 1e-4
        self.power = 0.5
        self.damping_factor = 0.2  # Factor for damping (per Saurav's suggestion)
        self.svd = svd
        self.inpainting = inpainting
        self.v_min = ((1 - self.alphas_cumprod) / self.alphas_cumprod)[0]
        self.mask = svd.mask.to(x_T.device)
        self.noise_sig_schedule = np.linspace(0.01, 0.5, 1000)
        self.Q = self.mask.shape[0]
        print(self.Q)

        self.betas = torch.tensor(betas).to(x_T.device)
        self.gamma_1 = 1e-6 * torch.ones(x_T.shape[0], self.Q, device=x_T.device)
        self.r_1 = (torch.sqrt(torch.tensor(1e-6)) * torch.randn_like(x_T)).to(x_T.device)
        self.r_2 = None
        self.gamma_2 = None

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        gamma_1_mult = torch.zeros(r_1.shape).to(y.device)
        for q in range(self.Q):
            gamma_1_mult += gamma_1[:, q, None, None, None] * self.mask[q, :, :, :]

        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        right_term = r_sig_inv * x_t
        right_term += self.svd.Ht(y).view(x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]) / noise_sig
        right_term += gamma_1_mult * r_1

        if self.Q > 1:  # Inpainting
            evals = (self.mask[0].unsqueeze(0).repeat(gamma_1.shape[0], 1, 1, 1) / noise_sig) ** 2
            inv_val = (evals + r_sig_inv ** 2 + gamma_1_mult) ** -1
            return inv_val * right_term, gamma_1_mult
        else:
            return self.svd.vamp_mu_1(right_term, noise_sig, r_sig_inv, gamma_1_mult).view(x_t.shape[0], x_t.shape[1],
                                                                                           x_t.shape[2],
                                                                                           x_t.shape[3]), gamma_1_mult

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig, gam1):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        singulars = self.svd.add_zeros(self.svd.singulars().unsqueeze(0).repeat(gamma_1.shape[0], 1))
        if self.Q > 1:  # Inpainting
            singulars = self.mask[0].unsqueeze(0).repeat(gamma_1.shape[0], 1, 1, 1)
        else:
            singulars = singulars.reshape(gamma_1.shape[0], -1).view(gamma_1.shape[0], 3, 256, 256)

        diag_mat_inv = ((singulars / noise_sig) ** 2 + r_sig_inv ** 2 + gamma_1) ** -1

        eta = torch.zeros(gamma_1.shape[0], self.Q).to(gamma_1.device)
        for q in range(self.Q):
            eta[:, q] += (diag_mat_inv * self.mask[q, None, :, :, :]).reshape(eta.shape[0], -1).sum(
                -1) / torch.count_nonzero(self.mask[q])

        return 1 / eta

    def uncond_denoiser_function(self, noisy_im, noise_var, t, t_alpha_bar):
        diff = torch.abs(
            noise_var[:, 0, None] - (1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(
                self.alphas_cumprod).to(noisy_im.device))
        t = torch.argmin(diff, dim=1)

        ones = torch.ones(noise_var.shape, device=noise_var.device)

        delta = torch.minimum(noise_var / self.v_min, ones)
        noise_var_clip = torch.maximum(noise_var, ones * self.v_min)

        # print(f'{noise_var[0].cpu().numpy()};{delta[0].cpu().numpy()};{t[0]}')
        scaled_noisy_im = noisy_im * torch.sqrt(1 / (1 + noise_var_clip[:, 0, None, None, None]))

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_est = torch.sqrt(noise_var_clip)[:, 0, None, None, None] * noise_predict
        x_0 = (1 - delta ** self.power)[:, 0, None, None, None] * noisy_im + (delta ** self.power)[:, 0, None, None,
                                                                             None] * (noisy_im - noise_est)
        # x_0 = noisy_im - torch.sqrt(noise_var_clip)[:, 0, None, None, None] * noise_predict

        return x_0,  ((1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(self.alphas_cumprod).to(noisy_im.device))[t]

    def denoiser_tr_approx(self, r_2, gamma_2, mu_2, t, t_alpha_bar, noise_var):
        eta = torch.zeros(gamma_2.shape).to(gamma_2.device)
        for k in range(self.K):
            # probe = torch.sign(torch.randn_like(mu_2).to(mu_2.device))
            probe = torch.randn_like(mu_2).to(r_2.device)
            # probe = probe / torch.norm(probe, dim=1, keepdim=True) # unit norm
            probe = probe / torch.sqrt(torch.mean(probe ** 2, dim=(1, 2, 3))[:, None, None, None])  # isotropic
            mu_2_delta, _ = self.uncond_denoiser_function((r_2 + self.delta * probe).float(), noise_var, t, t_alpha_bar)
            probed_diff = probe * (mu_2_delta - mu_2)

            for q in range(self.Q):
                masked_probe_diff = probed_diff * self.mask[q, None, :, :, :]
                eta[:, q] += masked_probe_diff.reshape(probed_diff.shape[0], -1).sum(-1) / (
                        self.delta * gamma_2[:, q] * torch.count_nonzero(self.mask[q]))

        return eta / self.K

    def linear_estimation(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        mu_1, gamma_1_mult = self.f_1(r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(gamma_1_mult, t_alpha_bar, noise_sig, gamma_1)

        gamma_2 = eta_1 - gamma_1
        r_2 = torch.zeros(mu_1.shape).to(mu_1.device)
        for q in range(self.Q):
            r_2 += ((eta_1[:, q, None, None, None] * mu_1 - gamma_1[:, q, None, None, None] * r_1) / gamma_2[:, q, None,
                                                                                                     None,
                                                                                                     None]) * self.mask[
                                                                                                              q, None,
                                                                                                              :, :, :]

        return mu_1, r_2, gamma_2, eta_1

    def denoising(self, r_2, gamma_2, t, t_alpha_bar):
        # Max var
        noise_var, _ = torch.max(1 / gamma_2, dim=1, keepdim=True)

        # Avg inv trace
        # noise_var = torch.zeros(gamma_2.shape[0], 1).to(gamma_2.device)
        # total_count = 0
        # for q in range(self.Q):
        #     total_count += torch.count_nonzero(self.mask[q])
        #     noise_var += torch.count_nonzero(self.mask[q]) / gamma_2[:, q]
        #
        # noise_var = noise_var / total_count

        # Denoise
        mu_2, true_noise_var = self.uncond_denoiser_function(r_2.float(), noise_var, t, t_alpha_bar)

        # print(noise_var)
        ################
        denoise_in = r_2.float()
        denoise_out = mu_2

        if t[0] % 1 == 0:
            plt.imsave(f'denoise_in.png', clear_color(denoise_in))
            plt.imsave(f'denoise_out.png', clear_color(denoise_out))

        # exit()

        ################

        tr = self.denoiser_tr_approx(r_2, gamma_2, mu_2, t, t_alpha_bar, noise_var)
        eta_2 = 1 / tr
        gamma_1 = eta_2 - gamma_2
        r_1 = torch.zeros(mu_2.shape).to(mu_2.device)
        for q in range(self.Q):
            r_1 += ((eta_2[:, q, None, None, None] * mu_2 - gamma_2[:, q, None, None, None] * r_2) / gamma_1[:, q, None,
                                                                                                     None,
                                                                                                     None]) * self.mask[
                                                                                                              q, None,
                                                                                                              :, :, :]

        return r_1, gamma_1, eta_2, mu_2, noise_var, true_noise_var.cpu().numpy()

    def run_vamp(self, x_t, y, t, noise_sig, use_damping=False):
        mu_2 = None  # needs to exist outside of for loop scope for return
        gamma_1 = self.gamma_1
        r_1 = self.r_1

        # noise_sig = self.noise_sig_schedule[t[0].cpu().numpy()]

        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        for i in range(1):
            old_gamma_1 = gamma_1
            old_r_1 = r_1

            _, r_2, gamma_2, eta_1 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig,
                                                         t_alpha_bar, noise_sig)

            # max_g_2, _ = torch.max(1/gamma_2, dim=1)

            # for q in range(self.Q):
            #     r_2 += (max_g_2 - 1/gamma_2[:, q]).sqrt() * torch.randn_like(r_2) * self.mask[q, None, :, :, :] # Noise measured region to missing level...

            # TODO: REMOVE...
            # r_2 += torch.randn_like(r_2) * ((1 - t_alpha_bar) / t_alpha_bar).sqrt()
            # gamma_2[:, 0] = t_alpha_bar / (1 - t_alpha_bar)

            r_1, gamma_1, eta_2, mu_2, noise_var, true_noise_var = self.denoising(r_2, gamma_2, t, t_alpha_bar)

            if use_damping:
                r_1 = self.damping_factor * r_1 + (1 - self.damping_factor) * old_r_1
                gamma_1 = (self.damping_factor * torch.abs(gamma_1) ** (-1 / 2) + (1 - self.damping_factor) * (
                    old_gamma_1) ** (-1 / 2)) ** -2

            print(f'eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()}; gamma_1 = {gamma_1[0].cpu().numpy()}; gamma_2 = {gamma_2[0].cpu().numpy()}; gamma_1 + gamma_2 = {(gamma_1 + gamma_2)[0].cpu().numpy()}')

            if torch.isnan(gamma_2).any(1).any(0) or torch.isnan(gamma_1).any(1).any(0):
                exit()

        self.gamma_1 = gamma_1
        self.r_1 = r_1

        return mu_2, gamma_1, gamma_2, eta_1, eta_2

    def run_vamp_reverse(self, x_t, y, t, noise_sig, use_damping=False):
        mu_1 = None  # needs to exist outside of for loop scope for return

        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        r_2 = x_t / torch.sqrt(t_alpha_bar)
        gamma_2 = torch.tensor([t_alpha_bar / (1 - t_alpha_bar)]*self.Q).unsqueeze(0).repeat(x_t.shape[0], 1).to(x_t.device)

        for i in range(3 if t[0] > 950 else 1):
            old_gamma_2 = gamma_2

            r_1, gamma_1, eta_2, mu_2, noise_var, true_noise_var = self.denoising(r_2, gamma_2, t, t_alpha_bar)
            mu_1, r_2, gamma_2, eta_1 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig,
                                                         t_alpha_bar, noise_sig)

            if use_damping:
                gamma_2_raw = gamma_2
                gamma_2 = (self.damping_factor * gamma_2_raw ** (-1 / 2) + (1 - self.damping_factor) * (
                    old_gamma_2) ** (-1 / 2)) ** -2

                new_r_2 = torch.zeros(r_2.shape).to(r_2.device)
                max_g_2, _ = torch.max(1/gamma_2, dim=1, keepdim=False)
                gam_diff = torch.maximum(max_g_2[:, None] - 1/gamma_2_raw, torch.zeros(gamma_2.shape).to(gamma_2.device))
                for q in range(self.Q):
                    new_r_2 += (r_2 + torch.randn_like(r_2).to(r_2.device) * gam_diff[:, q].sqrt()) * self.mask[q, None, :, :, :]

                r_2 = new_r_2

            print(f'eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()}; gamma_1 = {gamma_1[0].cpu().numpy()}; gamma_2 = {gamma_2[0].cpu().numpy()}; gamma_1 + gamma_2 = {(gamma_1 + gamma_2)[0].cpu().numpy()}')
            plt.imsave(f'mu_1.png', clear_color(mu_1))
            plt.imsave(f'mu_2.png', clear_color(mu_2))

            if torch.isnan(gamma_2).any(1).any(0) or torch.isnan(gamma_1).any(1).any(0):
                exit()

        return mu_1, gamma_1, gamma_2, eta_1, eta_2


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
