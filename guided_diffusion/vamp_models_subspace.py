import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from guided_diffusion.ddrm_svd import Deblurring

# TODO: Check if there is a bias...
# TODO: Plot MSE on the rs
# TODO: Verify singular values for colorization and super resolution

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

# TODO: Subspace VAMP, implement so when in subspace, DDRM transform is applied...
class VAMP:
    def __init__(self, model, betas, alphas_cumprod, max_iters, K, x_T, svd, inpainting=False):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.max_iters = max_iters
        self.K = 1
        self.delta = 1e-4
        self.power = 0.5
        self.damping_factor = 0.2  # Factor for damping (per Saurav's suggestion)
        self.damping_factor_g1 = 0.1
        self.damping_factors = np.flip(np.linspace(0.1, 0.5, 1000))
        self.svd = svd
        self.inpainting = inpainting
        self.v_min = ((1 - self.alphas_cumprod) / self.alphas_cumprod)[0]
        self.mask = svd.mask.to(x_T.device)
        self.noise_sig_schedule = np.linspace(0.01, 0.5, 1000)
        self.d = 3 * 256 * 256
        self.Q = 2 if self.d - self.svd.singulars().shape[0] > 0 else 1
        with open('eta_2_scale.npy', 'rb') as f:
            self.scale_factor = torch.from_numpy(np.load(f)).to(x_T.device)

        print(self.Q)

        self.betas = torch.tensor(betas).to(x_T.device)
        self.gamma_1 = 1e-6 * torch.ones(x_T.shape[0], self.Q, device=x_T.device)
        self.r_1 = (torch.sqrt(torch.tensor(1e-6)) * torch.randn_like(x_T)).to(x_T.device)
        self.r_2 = None
        self.gamma_2 = None

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        # gamma_1_mult = torch.zeros(r_1.shape).to(y.device)
        # for q in range(self.Q):
        #     gamma_1_mult += gamma_1[:, q, None, None, None] * self.mask[q, :, :, :]

        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        evals = (self.svd.singulars() / noise_sig) ** 2

        right_term = r_sig_inv * x_t
        right_term += self.svd.Ht(y).view(x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]) / noise_sig
        right_term = self.svd.Vt(right_term)

        scaled_r_1 = r_1
        scaled_r_1[:, :evals.shape[0]] = gamma_1[:, 0] * r_1[:, :evals.shape[0]]
        if self.Q > 1:
            scaled_r_1[:, evals.shape[0]:] = gamma_1[:, 1] * r_1[:, evals.shape[0]:]

        right_term += scaled_r_1

        nonzero_singular_mult = (evals[None, :] + r_sig_inv ** 2 + gamma_1[:, 0]) ** -1

        mu_1 = right_term
        mu_1[:, :evals.shape[0]] = nonzero_singular_mult * right_term[:, :evals.shape[0]]
        if self.Q > 1:
            mu_1[:, evals.shape[0]:] = (r_sig_inv ** 2 + gamma_1[:, 1]) ** -1 * mu_1[:, evals.shape[0]:]

        return mu_1


    def eta_1(self, t_alpha_bar, noise_sig, gamma_1):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        evals = (self.svd.singulars() / noise_sig) ** 2

        eta = torch.zeros(gamma_1.shape[0], self.Q).to(gamma_1.device)
        inv_measured = (evals[None, :] + r_sig_inv ** 2 + gamma_1[:, 0]) ** -1
        eta[:, 0] = inv_measured.mean(-1) ** -1
        if self.Q > 1:
            inv_nonmeasured = ((torch.ones(self.d - evals.shape[0]).to(gamma_1.device) * r_sig_inv ** 2)[None, :] + gamma_1[:, 1]) ** -1
            eta[:, 1] = inv_nonmeasured.mean(-1) ** -1

        return eta

    def uncond_denoiser_function(self, noisy_im, noise_var, gamma_2, noise=False):
        diff = torch.abs(
            noise_var[:, 0, None] - (1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(
                self.alphas_cumprod).to(noisy_im.device))
        t = torch.argmin(diff, dim=1)

        ones = torch.ones(noise_var.shape, device=noise_var.device)

        delta = torch.minimum(noise_var / self.v_min, ones)
        noise_var_clip = torch.maximum(noise_var, ones * self.v_min)

        scaled_noisy_im = noisy_im * torch.sqrt(1 / (1 + noise_var_clip[:, 0, None, None, None]))

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_est = torch.sqrt(noise_var_clip)[:, 0, None, None, None] * noise_predict
        x_0 = (1 - delta ** self.power)[:, 0, None, None, None] * noisy_im + (delta ** self.power)[:, 0, None, None,
                                                                             None] * (noisy_im - noise_est)

        return x_0, ((1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(
                self.alphas_cumprod).to(noisy_im.device))[t], t

    def linear_estimation(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig, gt=None):
        mu_1 = self.f_1(r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(t_alpha_bar, noise_sig, gamma_1)
        singulars = self.svd.singulars()

        gamma_2 = eta_1 - gamma_1

        max_g_2, _ = torch.max(1/gamma_2, dim=1)

        r_2 = torch.zeros(mu_1.shape).to(mu_1.device)
        noise = torch.randn_like(r_2)
        # noise = torch.zeros(mu_1.shape).to(mu_1.device)
        r_2[:, :singulars.shape[0]] = ((eta_1[:, 0, None] * mu_1 - gamma_1[:, 0, None] * r_1) / gamma_2[:, 0, None] + noise * (max_g_2 - 1/gamma_2[:, 0]).sqrt())[:, :singulars.shape[0]]
        if self.Q > 1:
            r_2[:, singulars.shape[0]:] = ((eta_1[:, 1, None] * mu_1 - gamma_1[:, 1, None] * r_1) / gamma_2[:, 1,None] + noise * (max_g_2 - 1/gamma_2[:, 1]).sqrt())[:, singulars.shape[0]:]

        # gamma_2 = 1/max_g_2.unsqueeze(1).repeat(1, self.Q)

        return mu_1, r_2, gamma_2, eta_1

    def get_eta_2(self, inv_gamma_2):
        diff = torch.abs(
            inv_gamma_2[:, 0, None] - (1 - torch.tensor(self.alphas_cumprod).to(inv_gamma_2.device)) / torch.tensor(
                self.alphas_cumprod).to(inv_gamma_2.device))
        t = torch.argmin(diff, dim=1)
        true_noise_var = ((1 - torch.tensor(self.alphas_cumprod).to(inv_gamma_2.device)) / torch.tensor(
                self.alphas_cumprod).to(inv_gamma_2.device))[t]

        eta_2 = torch.zeros(inv_gamma_2.shape).to(inv_gamma_2.device)
        eta_2[:, 0] = (self.scale_factor[t[0]] * true_noise_var.sqrt()).float()
        if self.Q > 1:
            diff = torch.abs(
                inv_gamma_2[:, 1, None] - (1 - torch.tensor(self.alphas_cumprod).to(inv_gamma_2.device)) / torch.tensor(
                    self.alphas_cumprod).to(inv_gamma_2.device))
            t = torch.argmin(diff, dim=1)
            true_noise_var = ((1 - torch.tensor(self.alphas_cumprod).to(inv_gamma_2.device)) / torch.tensor(
                self.alphas_cumprod).to(inv_gamma_2.device))[t]
            eta_2[:, 1] = (self.scale_factor[t[0]] * true_noise_var.sqrt()).float()

        return 1 / eta_2

    def denoising(self, r_2, gamma_2, t, vamp_iter=0, noise=False, gt=None):
        # Max var
        noise_var, _ = torch.max(1 / gamma_2, dim=1, keepdim=True)
        singulars = self.svd.singulars()

        new_r_2 = self.svd.V(r_2).view(r_2.shape[0], 3, 256, 256)

        # Denoise
        mu_2, true_noise_var, used_t = self.uncond_denoiser_function(new_r_2.float(), noise_var, gamma_2, noise)
        mu_2 = self.svd.Vt(mu_2)

        eta_2 = 1 / (self.scale_factor[used_t[0]] * true_noise_var.sqrt().repeat(r_2.shape[0], self.Q)).float()

        # eta_2 = self.get_eta_2(noise_var.repeat(1, self.Q))

        gamma_1 = eta_2 - gamma_2
        r_1 = torch.zeros(mu_2.shape).to(mu_2.device)
        r_1[:, :singulars.shape[0]] = ((eta_2[:, 0, None] * mu_2 - gamma_2[:, 0, None] * r_2) / gamma_1[:, 0, None])[:, :singulars.shape[0]]
        if self.Q > 1:
            r_1[:, singulars.shape[0]:] = ((eta_2[:, 1, None] * mu_2 - gamma_2[:, 1, None] * r_2) / gamma_1[:, 1, None])[:, singulars.shape[0]:]

        return r_1, gamma_1, eta_2, mu_2, noise_var, true_noise_var.cpu().numpy()

    def run_vamp_reverse_test(self, x_t, y, t, noise_sig, prob_name, gt, use_damping=False):
        mu_1 = None  # needs to exist outside of for loop scope for return
        singulars = self.svd.singulars()

        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        r_1 = self.svd.Vt(x_t / torch.sqrt(t_alpha_bar))
        r_2 = self.svd.Vt(x_t / torch.sqrt(t_alpha_bar))

        gamma_1 = torch.tensor([t_alpha_bar / (1 - t_alpha_bar)] * self.Q).unsqueeze(0).repeat(x_t.shape[0], 1).to(
            x_t.device)
        gamma_2 = torch.tensor([t_alpha_bar / (1 - t_alpha_bar)] * self.Q).unsqueeze(0).repeat(x_t.shape[0], 1).to(
            x_t.device)

        gam1s = []
        gam2s = []
        eta1s = []
        eta2s = []
        mu1s = []
        mu2s = []
        r1s = []
        r2s = []

        for i in range(100):
            old_gamma_1 = gamma_1.clone()
            old_gamma_2 = gamma_2.clone()

            old_r_1 = r_1.clone()
            old_r_2 = r_2.clone()

            plt.imsave(f'vamp_debug/{prob_name}/denoise_in/denoise_in_t={t[0].cpu().numpy()}_vamp_iter={i}.png', clear_color(self.svd.V(r_2).view(r_2.shape[0], 3, 256, 256)))

            r_1, gamma_1, eta_2, mu_2, noise_var, true_noise_var = self.denoising(r_2, gamma_2, t, vamp_iter=i, gt=gt)
            # if use_damping:
            #     damp_fac = self.damping_factor
            #
            #     if i > 1:
            #         gamma_1 = (damp_fac * gamma_1 ** (-1 / 2) + (1 - damp_fac) *
            #                    old_gamma_1 ** (-1 / 2)) ** -2
            #         r_1 = damp_fac * r_1 + (1 - damp_fac) * old_r_1

            mu_1, r_2, gamma_2, eta_1 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar),
                                                               y / noise_sig,
                                                               t_alpha_bar, noise_sig, gt=gt)


            plt.imsave(f'vamp_debug/{prob_name}/denoise_in_pre_damp/denoise_in_t={t[0].cpu().numpy()}_vamp_iter={i}.png', clear_color(self.svd.V(r_2).view(r_2.shape[0], 3, 256, 256)))

            if use_damping:
                damp_fac = self.damping_factor

                # gamma_2_raw = gamma_2.clone().abs()
                # gamma_2 = (damp_fac * gamma_2_raw ** (-1 / 2) + (1 - damp_fac) * old_gamma_2 ** (-1 / 2)) ** -2
                # r_2[:, :singulars.shape[0]] = (r_2 + torch.randn_like(r_2).to(r_2.device) * torch.maximum((1 / gamma_2 - 1 / gamma_2_raw), torch.zeros(gamma_2.shape).to(gamma_2.device)).sqrt()[:, 0])[:, :singulars.shape[0]]
                # if self.Q > 1:
                #     r_2[:, singulars.shape[0]:] = (r_2 + torch.randn_like(r_2).to(r_2.device) * torch.maximum(
                #         (1 / gamma_2 - 1 / gamma_2_raw), torch.zeros(gamma_2.shape).to(gamma_2.device)).sqrt()[:, 1])[:,
                #                                   singulars.shape[0]:]

                gamma_2 = (damp_fac * gamma_2 ** (-1 / 2) + (1 - damp_fac) *
                           old_gamma_2 ** (-1 / 2)) ** -2
                r_2 = damp_fac * r_2 + (1 - damp_fac) * old_r_2

            if torch.linalg.norm(mu_1 - mu_2).cpu().numpy() > 1e4:
                break

            eta1s.append(1/eta_1[0, 0].cpu().numpy())
            eta2s.append(1/eta_2[0, 0].cpu().numpy())
            gam1s.append(1/gamma_1[0, 0].cpu().numpy())
            gam2s.append(1/gamma_2[0, 0].cpu().numpy())
            mu1s.append(self.svd.V(mu_1).view(r_2.shape[0], 3, 256, 256))
            mu2s.append(self.svd.V(mu_2).view(r_2.shape[0], 3, 256, 256))
            r1s.append(self.svd.V(r_1).view(r_2.shape[0], 3, 256, 256))
            r2s.append(self.svd.V(r_1).view(r_2.shape[0], 3, 256, 256))

            plt.imsave(f'vamp_debug/{prob_name}/mu_1_v_step/mu_1_t={t[0].cpu().numpy()}_vamp_iter={i}.png', clear_color(self.svd.V(mu_1).view(r_2.shape[0], 3, 256, 256)))
            plt.imsave(f'vamp_debug/{prob_name}/mu_2_v_step/mu_2_t={t[0].cpu().numpy()}_vamp_iter={i}.png', clear_color(self.svd.V(mu_2).view(r_2.shape[0], 3, 256, 256)))
            plt.imsave(f'vamp_debug/{prob_name}/r1s/r_1_t={t[0].cpu().numpy()}_vamp_iter={i}.png', clear_color(self.svd.V(r_2).view(r_2.shape[0], 3, 256, 256)))


            print(
                f'||mu_1 - mu_2|| = {torch.linalg.norm(mu_1 - mu_2).cpu().numpy()}; eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()}; gamma_1 = {gamma_1[0].cpu().numpy()}; gamma_2 = {gamma_2[0].cpu().numpy()}; gamma_1 + gamma_2 = {(gamma_1 + gamma_2)[0].cpu().numpy()}')

            # time.sleep(30)

        return_val = mu_1
        return return_val, eta1s, eta2s, gam1s, gam2s, mu1s, mu2s, r1s, r2s

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
