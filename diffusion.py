from tqdm import tqdm

import numpy as np
import torch


# https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/train.py#L120
class EMA:
    """
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
            
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/train.py#L154
class Diffusion:
    def __init__(self, image_size, num_image_channels, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device=None):
        self.device = device

        self.image_size = image_size
        self.num_image_channels = num_image_channels
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    @torch.no_grad()
    def sampling(self, unet, chars, writers_idx, mix_rate=None, cfg_scale=3):
        unet.eval()
        
        n = len(chars)
        
        if writers_idx is None:
            pass
        else:
            assert n == writers_idx.shape[0]

        with torch.no_grad():
            x = torch.randn((n, self.num_image_channels, self.image_size, self.image_size), device=self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i
                predicted_noise = unet(x, t, chars, writers_idx, mix_rate=mix_rate)
                
                if cfg_scale > 0:
                    uncond_predicted_noise = unet(x, t, chars, writers_idx, mix_rate)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        unet.train()
        
        return x
