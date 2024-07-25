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
        self.max_iters = 50
        self.K = 1
        self.delta = 1e-4
        self.power = 0.5
        self.damping_factor = 0.75  # Factor for damping (per Saurav's suggestion)
        self.damping_factor_g1 = 0.1
        self.damping_factors = np.flip(np.linspace(0.1, 0.5, 1000))
        self.svd = svd
        self.inpainting = inpainting
        self.v_min = ((1 - self.alphas_cumprod) / self.alphas_cumprod)[0]
        self.mask = svd.mask.to(x_T.device)
        self.noise_sig_schedule = np.linspace(0.01, 0.5, 1000)
        self.rho = 2
        self.xi = 1/25
        self.d = 3 * 256 * 256
        self.Q = 2 if self.d - self.svd.singulars().shape[0] > 0 else 1
        with open('eta_2_scale.npy', 'rb') as f:
            self.scale_factor = torch.from_numpy(np.load(f)).to(x_T.device)

        print(self.Q)
        print('SUBSPACE!!!')

        self.betas = torch.tensor(betas).to(x_T.device)
        self.mu_2 = None
        self.eta_2 = None
        self.gamma_2 = None
        self.nfes = 0

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        evals = (self.svd.singulars() / noise_sig) ** 2

        right_term = r_sig_inv * x_t
        right_term += self.svd.Ht(y).view(x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3]) / noise_sig
        right_term = self.svd.Vt(right_term)

        scaled_r_1 = torch.zeros(right_term.shape).to(right_term.device)
        scaled_r_1[:, :evals.shape[0]] = gamma_1[:, 0] * r_1[:, :evals.shape[0]]
        if self.Q > 1:
            scaled_r_1[:, evals.shape[0]:] = gamma_1[:, 1] * r_1[:, evals.shape[0]:]

        right_term += scaled_r_1

        nonzero_singular_mult = (evals[None, :] + r_sig_inv ** 2 + gamma_1[:, 0]) ** -1

        mu_1 = torch.zeros(right_term.shape).to(right_term.device)
        mu_1[:, :evals.shape[0]] = nonzero_singular_mult * right_term[:, :evals.shape[0]]
        if self.Q > 1:
            mu_1[:, evals.shape[0]:] = (r_sig_inv ** 2 + gamma_1[:, 1]) ** -1 * right_term[:, evals.shape[0]:]

        return mu_1

    def eta_1(self, t_alpha_bar, noise_sig, gamma_1):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        evals = (self.svd.singulars() / noise_sig) ** 2

        eta = torch.zeros(gamma_1.shape[0], self.Q).to(gamma_1.device)
        inv_measured = (evals[None, :] + r_sig_inv ** 2 + gamma_1[:, 0]) ** -1
        eta[:, 0] = inv_measured.mean(-1) ** -1
        if self.Q > 1:
            eta[:, 1] = r_sig_inv ** 2 + gamma_1[:, 1]
            # new_evals = (self.svd.add_zeros(evals.unsqueeze(0).repeat(gamma_1.shape[0], 1)) + r_sig_inv ** 2 + gamma_1[:, 1]) ** -1
            # eta[:, 0] = new_evals.mean(-1) ** -1
            # eta[:, 1] = new_evals.mean(-1) ** -1

        return eta

    def uncond_denoiser_function(self, noisy_im, noise_var):
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

    def linear_estimation(self, mu_2, eta_2, x_t, y, t_alpha_bar, noise_sig):
        mu_1 = self.f_1(mu_2, eta_2, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(t_alpha_bar, noise_sig, eta_2)

        return mu_1, eta_1

    def denoising(self, mu_1, gamma_2):
        # Max var
        noise_var, _ = torch.max(1 / gamma_2, dim=1, keepdim=True)

        new_mu_1 = self.svd.V(mu_1).view(mu_1.shape[0], 3, 256, 256)

        # Denoise
        mu_2, true_noise_var, used_t = self.uncond_denoiser_function(new_mu_1.float(), noise_var)
        mu_2 = self.svd.Vt(mu_2)

        eta_2 = 1 / (self.scale_factor[used_t[0]] * true_noise_var.sqrt().repeat(1, self.Q)).float()

        return mu_2, eta_2

    def run_vamp_reverse_test(self, x_t, y, t, noise_sig, prob_name, gt, use_damping=False):
        singulars = self.svd.singulars()
        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        # 0. Initialize Values
        if t[0] % 1 == 0: # Occasional cold start
            self.mu_2 = None
            self.eta_2 = None
            self.gamma_2 = None

        mu_2 = self.mu_2
        eta_2 = self.eta_2
        gamma_2 = self.gamma_2

        if mu_2 is None:
            mu_2 = self.svd.Vt(x_t / torch.sqrt(t_alpha_bar))
            eta_2 = torch.zeros(x_t.shape[0], 2).to(x_t.device)
            gamma_2 = torch.tensor([t_alpha_bar / (1 - t_alpha_bar)]).unsqueeze(0).repeat(x_t.shape[0], 1).to(x_t.device) / 2

        gamma2s = []
        eta1s = [[], []]
        eta2s = [[], []]
        mu1s = [[], []]
        mu2s = [[], []]

        for i in range(self.max_iters):
            # plt.imsave(
            #     f'vamp_debug/{prob_name}/posterior/denoise_in/denoise_in_t={t[0].cpu().numpy()}_vamp_iter={i}.png',
            #     clear_color(self.svd.V(mu_1_noised).view(mu_1_noised.shape[0], 3, 256, 256)))

            # 1. Linear Estimation
            mu_1, eta_1 = self.linear_estimation(mu_2, eta_2, x_t / torch.sqrt(1 - t_alpha_bar),
                                                 y / noise_sig,
                                                 t_alpha_bar, noise_sig)
            # 2. Re-Noising
            noise = torch.randn_like(mu_1)
            zeros = torch.zeros(mu_1.shape).to(mu_1.device)

            old_gamma_2 = gamma_2.clone()
            gamma_2 = self.rho * gamma_2
            mean_eta_1 = singulars.shape[0] / eta_1[:, 0]
            if self.Q > 1:
                mean_eta_1 += (self.d - singulars.shape[0]) / eta_1[:, 1]

            mean_eta_1 = mean_eta_1 / self.d
            if (gamma_2 > self.xi / mean_eta_1).any() and self.eta_2 is not None:
                gamma_2 = old_gamma_2
                break

            v_1_measured = 1 / gamma_2 - 1 / eta_1[:, 0]
            v_1_measured = torch.maximum(v_1_measured, zeros)
            mu_1_noised = torch.zeros(mu_1.shape).to(mu_1.device)
            mu_1_noised[:, :singulars.shape[0]] = (mu_1 + noise * v_1_measured.sqrt())[:, :singulars.shape[0]]
            if self.Q > 1:
                v_1_nonmeasured = 1 / gamma_2 - 1 / eta_1[:, 1]
                v_1_measured = torch.maximum(v_1_nonmeasured, zeros)
                mu_1_noised[:, singulars.shape[0]:] = (mu_1 + noise * v_1_measured.sqrt())[:, singulars.shape[0]:]

            # 3. Denoising
            mu_2, eta_2 = self.denoising(mu_1_noised, gamma_2)
            self.nfes += 1
            self.mu_2 = mu_2
            self.eta_2 = eta_2
            self.gamma_2 = gamma_2

            eta1s[0].append(1 / eta_1[0, 0].cpu().numpy())
            eta2s[0].append(1 / eta_2[0, 0].cpu().numpy())
            mu1s[0].append(self.svd.V(mu_1).view(mu_1.shape[0], 3, 256, 256))
            mu2s[0].append(self.svd.V(mu_2).view(mu_1.shape[0], 3, 256, 256))
            gamma2s.append(1 / gamma_2[0].cpu().numpy())

            if self.Q > 1:
                eta1s[1].append(1 / eta_1[0, 1].cpu().numpy())
                eta2s[1].append(1 / eta_2[0, 1].cpu().numpy())
                mu1s[1].append(self.svd.V(mu_1).view(mu_1.shape[0], 3, 256, 256))
                mu2s[1].append(self.svd.V(mu_2).view(mu_1.shape[0], 3, 256, 256))

            # plt.imsave(f'vamp_debug/{prob_name}/posterior/mu_1_v_step/mu_1_t={t[0].cpu().numpy()}_vamp_iter={i}.png',
            #            clear_color(self.svd.V(mu_1).view(mu_1.shape[0], 3, 256, 256)))
            # plt.imsave(f'vamp_debug/{prob_name}/posterior/mu_2_v_step/mu_2_t={t[0].cpu().numpy()}_vamp_iter={i}.png',
            #            clear_color(self.svd.V(mu_2).view(mu_1.shape[0], 3, 256, 256)))

            print(
                f'ITER: {i + 1}; gamma_2 = {gamma_2[0].cpu().numpy()}; ||mu_1 - mu_2|| = {torch.linalg.norm(mu_1 - mu_2).cpu().numpy()}; eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()};\n')


        return_val = self.svd.V(mu_2).view(mu_2.shape[0], 3, 256, 256)
        print(self.nfes)

        return return_val, eta1s, eta2s, mu1s, mu2s, gamma2s


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
