import torch

class VAMP:
    def __init__(self, model, betas, alphas_cumprod, max_iters, K, x_T):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.max_iters = 5
        self.K = K
        self.delta = 1e-4
        self.damping_factor = 0.5 # Factor for damping (per Saurav's suggestion)

        self.betas = torch.tensor(betas).to(x_T.device)
        self.gamma_1 = 1e-6 * torch.ones(x_T.shape[0], 1, device=x_T.device)
        self.r_1 = (torch.sqrt(torch.tensor(1e-6)) * torch.randn_like(x_T)).to(x_T.device)
        self.gamma_2 = None
        self.r_2 = None
        # self.gamma_1 = torch.ones(x_T.shape[0], 1, device=x_T.device)
        # self.r_1 = torch.randn_like(x_T).to(x_T.device)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        raise NotImplementedError()

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        raise NotImplementedError()

    def uncond_denoiser_function(self, noisy_im, noise_var, t, t_alpha_bar):
        scale_factor_prime = torch.sqrt((1 - t_alpha_bar) / noise_var)
        scale_factor = scale_factor_prime / torch.sqrt(t_alpha_bar)

        scaled_noisy_im = noisy_im * scale_factor_prime[:, 0, None, None, None]

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        x_0_scaled = (scaled_noisy_im - torch.sqrt(1 - t_alpha_bar) * noise_predict) / (torch.sqrt(t_alpha_bar) * scale_factor)

        return x_0_scaled

    def denoiser_tr_approx(self, r_2, gamma_2, mu_2, t, t_alpha_bar):
        tr_out = torch.zeros(mu_2.shape[0], 1).to(mu_2.device)
        for k in range(self.K):
            probe = torch.sign(torch.randn_like(mu_2).to(mu_2.device))
            mu_2_delta = self.uncond_denoiser_function((r_2 + self.delta * probe).float(), 1 / gamma_2, t, t_alpha_bar)

            tr_out += torch.mean((probe * (mu_2_delta - mu_2)).view(mu_2.shape[0], -1), 1, keepdim=True) / self.delta

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
        r_2 = self.r_2  # needs to exist outside of for loop scope for damping
        gamma_2 = self.gamma_2  # needs to exist outside of for loop scope for damping
        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        for i in range(self.max_iters):
            # Keep history for damping
            old_r_1 = r_1
            old_r_2 = r_2

            old_gamma_1 = gamma_1
            old_gamma_2 = gamma_2

            r_2, gamma_2, eta_1 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig, t_alpha_bar, noise_sig)
            r_1, gamma_1, eta_2, mu_2 = self.denoising(r_2, gamma_2, t, t_alpha_bar)

            print(f'eta_1 = {eta_1[0].cpu().numpy()}; eta_2 = {eta_2[0].cpu().numpy()}; gamma_1 = {gamma_1[0].cpu().numpy()}; gamma_2 = {gamma_2[0].cpu().numpy()}; gamma_1 + gamma_2 = {(gamma_1 + gamma_2)[0].cpu().numpy()}')

            if torch.isnan(gamma_2) or torch.isnan(gamma_1):
                exit()

            # Damping - damp both gammas and both rs
            if use_damping and t[0] < 999:
                r_1 = self.damping_factor * r_1 + (1 - self.damping_factor) * old_r_1
                r_2 = self.damping_factor * r_2 + (1 - self.damping_factor) * old_r_2

                gamma_1 = (self.damping_factor * torch.abs(gamma_1) ** (-1 / 2) + (1 - self.damping_factor) * (
                    old_gamma_1) ** (-1 / 2)) ** -2
                gamma_2 = (self.damping_factor * torch.abs(gamma_2) ** (-1 / 2) + (1 - self.damping_factor) * (
                    old_gamma_2) ** (-1 / 2)) ** -2

        self.gamma_1 = gamma_1
        self.r_1 = r_1
        self.gamma_2 = gamma_2
        self.r_2 = r_2

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


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)