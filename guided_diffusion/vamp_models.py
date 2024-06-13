import torch
from guided_diffusion.ddrm_svd import Deblurring


class VAMP:
    def __init__(self, model, betas, alphas_cumprod, max_iters, K, x_T, svd, inpainting=False):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.max_iters = max_iters
        self.K = K
        self.delta = 1e-4
        self.damping_factor = 1 # Factor for damping (per Saurav's suggestion)
        self.svd = svd
        self.inpainting = inpainting

        self.betas = torch.tensor(betas).to(x_T.device)
        self.gamma_1 = 1e-6 * torch.ones(x_T.shape[0], 1, device=x_T.device)
        self.r_1 = (torch.sqrt(torch.tensor(1e-6)) * torch.randn_like(x_T)).to(x_T.device)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        right_term = r_sig_inv * x_t
        right_term += 1 / noise_sig * self.svd.Ht(y).view(x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3])
        right_term += gamma_1[:, 0, None, None, None] * r_1

        return self.svd.vamp_mu_1(right_term, noise_sig, r_sig_inv, gamma_1).view(x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3])

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        singulars = self.svd.add_zeros(self.svd.singulars().unsqueeze(0).repeat(gamma_1.shape[0], 1))
        eta = torch.mean(((singulars / noise_sig) ** 2 + r_sig_inv ** 2 + gamma_1[:, 0, None]) ** -1, dim=1,
                         keepdim=True) ** -1

        return eta

    def uncond_denoiser_function(self, noisy_im, noise_var, t, t_alpha_bar):
        diff = torch.abs(noise_var[:, 0, None] - (1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(self.alphas_cumprod).to(noisy_im.device))
        nearest_indices = torch.argmin(diff, dim=1)

        v_min = ((1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(self.alphas_cumprod).to(noisy_im.device))[0].float()

        delta = noise_var.clone() / v_min
        delta[delta > 1] = 1.
        noise_var[delta < 1] = v_min

        t = nearest_indices
        scaled_noisy_im = noisy_im * torch.sqrt(1 / (1 + noise_var[:, 0, None, None, None]))

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_to_remove = ((delta ** 0.25) * torch.sqrt(noise_var))[:, 0, None, None, None] * noise_predict
        x_0 = noisy_im - noise_to_remove

        # x_0 = noisy_im - torch.sqrt(noise_var)[:, 0, None, None, None] * noise_predict

        return x_0

    def denoiser_tr_approx(self, r_2, gamma_2, mu_2, t, t_alpha_bar):
        tr_out = torch.zeros(mu_2.shape[0], 1).to(mu_2.device)
        for k in range(self.K):
            # probe = torch.sign(torch.randn_like(mu_2).to(mu_2.device))
            probe = torch.randn_like(mu_2).to(r_2.device)
            # probe = probe / torch.norm(probe, dim=1, keepdim=True) # unit norm
            probe = probe / torch.sqrt(torch.mean(probe ** 2, dim=(1, 2, 3))[:, None, None, None])  # isotropic
            mu_2_delta = self.uncond_denoiser_function((r_2 + self.delta * probe).float(), 1 / gamma_2, t, t_alpha_bar)

            tr_out += torch.mean((probe * (mu_2_delta - mu_2)).reshape(mu_2.shape[0], -1), 1, keepdim=True) / self.delta

        return tr_out / self.K

    def linear_estimation(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        mu_1 = self.f_1(r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(gamma_1, t_alpha_bar, noise_sig)

        gamma_2 = eta_1 - gamma_1
        r_2 = (eta_1[:, 0, None, None, None] * mu_1 - gamma_1[:, 0, None, None, None] * r_1) / gamma_2[:, 0, None, None, None]

        return r_2, gamma_2, eta_1

    def denoising(self, r_2, gamma_2, t, t_alpha_bar):
        mu_2 = self.uncond_denoiser_function(r_2.float(), 1 / gamma_2, t, t_alpha_bar)
        eta_2 = gamma_2 / self.denoiser_tr_approx(r_2, gamma_2, mu_2, t, t_alpha_bar)
        gamma_1 = eta_2 - gamma_2
        r_1 = (eta_2[:, 0, None, None, None] * mu_2 - gamma_2[:, 0, None, None, None] * r_2) / gamma_1[:, 0, None, None, None]

        return r_1, gamma_1, eta_2, mu_2

    def run_vamp(self, x_t, y, t, noise_sig, use_damping=False):
        mu_2 = None  # needs to exist outside of for loop scope for return
        gamma_1 = self.gamma_1
        r_1 = self.r_1

        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        for i in range(2):
            old_gamma_1 = gamma_1
            old_r_1 = r_1

            r_2, gamma_2, eta_1 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig, t_alpha_bar, noise_sig)
            r_1, gamma_1, eta_2, mu_2 = self.denoising(r_2, gamma_2, t, t_alpha_bar)

            if use_damping:
                r_1 = self.damping_factor * r_1 + (1 - self.damping_factor) * old_r_1
                gamma_1 = (self.damping_factor * torch.abs(gamma_1) ** (-1 / 2) + (1 - self.damping_factor) * (
                    old_gamma_1) ** (-1 / 2)) ** -2

            print(f'eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()}; gamma_1 = {gamma_1[0].cpu().numpy()}; gamma_2 = {gamma_2[0].cpu().numpy()}; gamma_1 + gamma_2 = {(gamma_1 + gamma_2)[0].cpu().numpy()}')

            if torch.isnan(gamma_2).any(1).any(0) or torch.isnan(gamma_1).any(1).any(0):
                exit()

        self.gamma_1 = gamma_1 if use_damping and self.damping_factor < 1 else 0.2 * gamma_1
        self.r_1 = r_1

        return mu_2


class Denoising(VAMP):
    def __init__(self, model, betas, alphas_cumprod, max_iters, x_T, K=1):
        super().__init__(model, betas, alphas_cumprod, max_iters, K, x_T)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        return 1 / (1 / (noise_sig ** 2) + (r_sig_inv ** 2) + gamma_1[:, 0, None, None, None]) * (r_sig_inv * x_t + (1 / (noise_sig)) * y + gamma_1[:, 0, None, None, None] * r_1)

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        return r_sig_inv ** 2 + 1 / (noise_sig ** 2) + gamma_1


class Inpainting(VAMP):
    def __init__(self, model, betas, alphas_cumprod, max_iters, x_T, kept_ones, missing_ones, K=1):
        super().__init__(model, betas, alphas_cumprod, max_iters, K, x_T)
        self.kept_ones = kept_ones
        self.missing_ones = missing_ones

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        right_term = r_sig_inv * x_t + (1 / noise_sig) * y * self.kept_ones + gamma_1[:, 0, None, None, None] * r_1

        kept_ones = 1 / (1 / (noise_sig ** 2) + (r_sig_inv ** 2) + gamma_1[:, 0, None, None, None]) * self.kept_ones
        missing_ones = 1 / ((r_sig_inv ** 2) + gamma_1[:, 0, None, None, None]) * self.missing_ones

        return right_term * (kept_ones + missing_ones)

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        total_missing = torch.count_nonzero(self.missing_ones, dim=(1, 2, 3))
        total_kept = torch.count_nonzero(self.kept_ones, (1, 2, 3))

        sum_1 = total_missing[:, None] * ((r_sig_inv ** 2 + gamma_1) ** -1)
        sum_2 = total_kept[:, None] * ((1 / (noise_sig ** 2) + r_sig_inv ** 2 + gamma_1) ** -1)

        return ((sum_1 + sum_2) / (total_kept[:, None] + total_missing[:, None])) ** -1


class Deblur(VAMP):
    def __init__(self, model, betas, alphas_cumprod, max_iters, x_T, kernel, K=1):
        super().__init__(model, betas, alphas_cumprod, max_iters, K, x_T)
        self.deblur_svd = Deblurring(kernel, x_T.shape[1], x_T.shape[2], x_T.device)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        right_term = r_sig_inv * x_t
        right_term += 1 / noise_sig * self.deblur_svd.Ht(y).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3])
        right_term += gamma_1[:, 0, None, None, None] * r_1

        return self.deblur_svd.vamp_mu_1(right_term, noise_sig, r_sig_inv, gamma_1).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3])

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        singulars = self.deblur_svd.add_zeros(self.deblur_svd.singulars())
        eta = torch.mean(((singulars / noise_sig) ** 2 + r_sig_inv ** 2 + gamma_1[None, :, 0]) ** -1, dim=0, keepdim=True) ** -1

        return eta


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)