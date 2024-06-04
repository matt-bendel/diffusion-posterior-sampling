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
        diff = torch.abs(noise_var - self.betas)
        nearest_indices = torch.argmin(diff, dim=1)

        t = nearest_indices.repeat(noisy_im.shape[0])
        noise_predict = self.model(noisy_im, t)

        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, 0)[nearest_indices]

        print(alphas_cumprod)
        print(noisy_im.shape)
        print(noise_predict.shape)
        exit()

        x_0 = noisy_im / torch.sqrt(alphas_cumprod) - torch.sqrt((1 - alphas_cumprod) / alphas_cumprod) * noise_predict

        return x_0

    def denoiser_tr_approx(self, r_2, gamma_2, mu_2):
        tr_out = torch.zeros(mu_2.shape[0], 1)
        for k in range(self.K):
            probe = torch.sign(torch.randn_like(mu_2).to(mu_2.device))
            mu_2_delta = self.uncond_denoiser_function(r_2 + self.delta * probe, 1 / gamma_2)

            tr_out += torch.mean((probe * (mu_2_delta - mu_2)).view(mu_2.shape[0], -1), 1) / self.delta

        return tr_out / self.K

    def linear_estimation(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        mu_1 = self.f_1(r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig)
        eta_1 = self.eta_1(gamma_1, t_alpha_bar, noise_sig)
        gamma_2 = eta_1 - gamma_1
        r_2 = (eta_1 * mu_1 - gamma_1 * r_1) / gamma_2

        return r_2, gamma_2

    def denoising(self, r_2, gamma_2):
        mu_2 = self.uncond_denoiser_function(r_2, 1 / gamma_2)
        eta_2 = gamma_2 / self.denoiser_tr_approx(r_2, gamma_2, mu_2)
        gamma_1 = eta_2 - gamma_2
        r_1 = (eta_2 * mu_2 - gamma_2 * r_2) / gamma_1

        return r_1, gamma_1, mu_2

    def run_vamp(self, x_t, y, t, noise_sig):
        mu_2 = None
        gamma_1 = self.gamma_1
        r_1 = self.r_1
        t_alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_t)[0, 0, 0, 0]

        for i in range(self.max_iters):
            r_2, gamma_2 = self.linear_estimation(r_1, gamma_1, x_t / torch.sqrt(1 - t_alpha_bar), y / noise_sig, t_alpha_bar, noise_sig)
            r_1, gamma_1, mu_2 = self.denoising(r_2, gamma_2)

        self.gamma_1 = gamma_1
        self.r_1 = r_1

        return mu_2


class Denoising(VAMP):
    def __init__(self, model, betas, alphas_cumprod, max_iters, x_T, K=1):
        super().__init__(model, betas, alphas_cumprod, max_iters, K, x_T)

    def f_1(self, r_1, gamma_1, x_t, y, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        return (1 / (noise_sig ** 2) + (r_sig_inv ** 2) + gamma_1) * (r_sig_inv ** 2 * x_t + (1 / (noise_sig ** 2)) * y + gamma_1 * r_1)

    def eta_1(self, gamma_1, t_alpha_bar, noise_sig):
        r_sig_inv = torch.sqrt(t_alpha_bar / (1 - t_alpha_bar))
        return r_sig_inv ** 2 + 1 / (noise_sig ** 2) + gamma_1


# class Inpainting(VAMP):
#     def __init__(self, model, betas, alphas_cumprod, max_iters, K=1):
#         super().__init__(model, betas, alphas_cumprod, max_iters, K)
#
#     def f_1(self, r_1, gamma_1, x_t, y, noise_sig):
#         raise NotImplementedError() # TODO:
#
#     def eta_1(self, gamma_1, x_t, y, noise_sig):
#         raise NotImplementedError() # TODO


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)