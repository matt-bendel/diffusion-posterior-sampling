import torch

class VAMP:
    def __init__(self, model, betas, alphas_cumprod, max_iters, K, x_T):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.max_iters = max_iters
        self.K = K
        self.delta = 1e-4
        self.betas = torch.tensor(betas).to(x_T.device)
        self.gamma_1 = 1e-6 * torch.ones(x_T.shape[0], 1, device=x_T.device)
        self.r_1 = (torch.sqrt(torch.tensor(1e-6)) * torch.randn_like(x_T)).to(x_T.device)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        raise NotImplementedError()

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        raise NotImplementedError()

    def uncond_denoiser_function(self, noisy_im, noise_var):
        diff = torch.abs(noise_var - (1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)))
        nearest_indices = torch.argmin(diff, dim=1)

        t = nearest_indices.repeat(noisy_im.shape[0])
        noise_predict = self.model(noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        x_0 = (noisy_im - torch.sqrt(noise_var) * noise_predict) / torch.sqrt(1 - noise_var)

        return x_0

    def denoiser_tr_approx(self, r_2, gamma_2, mu_2):
        tr_out = torch.zeros(mu_2.shape[0], 1).to(mu_2.device)
        for k in range(self.K):
            probe = torch.sign(torch.randn_like(mu_2).to(mu_2.device))
            mu_2_delta = self.uncond_denoiser_function((r_2 + self.delta * probe).float(), 1 / gamma_2)

            tr_out += torch.mean((probe * (mu_2_delta - mu_2)).view(mu_2.shape[0], -1), 1) / self.delta

        return tr_out / self.K

    def linear_estimation(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        mu_1 = self.f_1(r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(gamma_1, t_alpha_bar, noise_sig)
        gamma_2 = eta_1 - gamma_1
        r_2 = (eta_1 * mu_1 - gamma_1 * r_1) / gamma_2

        return r_2.detach(), gamma_2.detach()

    def denoising(self, r_2, gamma_2):
        mu_2 = self.uncond_denoiser_function(r_2.float(), 1 / gamma_2)
        eta_2 = gamma_2 / self.denoiser_tr_approx(r_2, gamma_2, mu_2)
        gamma_1 = eta_2 - gamma_2
        r_1 = (eta_2 * mu_2 - gamma_2 * r_2) / gamma_1

        return r_1.detach(), gamma_1.detach(), mu_2.detach()

    def run_vamp(self, x_t, y, t, noise_sig):
        mu_2 = None
        gamma_1 = self.gamma_1
        r_1 = self.r_1
        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        for i in range(self.max_iters):
            r_2, gamma_2 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig, t_alpha_bar, noise_sig)
            r_1, gamma_1, mu_2 = self.denoising(r_2, gamma_2)

            if torch.isnan(gamma_2) or torch.isnan(gamma_1):
                exit()

        self.gamma_1 = gamma_1.detach()
        self.r_1 = r_1.detach()

        return mu_2.detach()


class Denoising(VAMP):
    def __init__(self, model, betas, alphas_cumprod, max_iters, x_T, K=1):
        super().__init__(model, betas, alphas_cumprod, max_iters, K, x_T)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        return 1 / (1 / (noise_sig ** 2) + (r_sig_inv ** 2) + gamma_1) * (r_sig_inv * x_t + (1 / (noise_sig)) * y + gamma_1 * r_1)

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

        right_term = r_sig_inv * x_t + (1 / noise_sig) * y + gamma_1 * r_1

        kept_ones = self.kept_ones * (1 / (r_sig_inv ** 2 + 1 / (noise_sig ** 2) + gamma_1))
        missing_ones = self.missing_ones * (1 / (r_sig_inv ** 2 + gamma_1))

        return right_term * (kept_ones + missing_ones)

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))

        total_missing = torch.count_nonzero(self.missing_ones)
        total_kept = torch.count_nonzero(self.kept_ones)

        print(total_kept)
        print(total_missing)

        sum_1 = total_missing * ((1 / (noise_sig ** 2) + gamma_1) ** -1)
        sum_2 = total_kept * ((1 / (noise_sig ** 2) + r_sig_inv ** 2 + gamma_1) ** -1)

        return ((sum_1 + sum_2) / (total_kept + total_missing)) ** -1


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)