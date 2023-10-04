import copy
import json
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "stable_diffusion"))

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from diffusers import AutoencoderKL

from dataset import ConcatDataset
from layer import UNetModel
from model import EMA, Diffusion
from utility import char2code, save_images


STABLE_DIFFUSION_CHANNEL = 4


class RadicalStylist:
    def __init__(
        self,
        
        save_path,
        stable_diffusion_path,
        
        radicalname2idx,
        writer2idx,
        
        image_size,
        dim_char_embedding,
        char_length,
        num_res_blocks,
        num_heads,
        
        learning_rate,
        ema_beta,
        diffusion_noise_steps,
        diffusion_beta_start,
        diffusion_beta_end,
        
        diversity_lambda,
        
        device,
    ):
        self.device = device
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        
        self.radicalname2idx = radicalname2idx
        with open(os.path.join(self.save_path, "radicalname2idx.json"), "w") as f:
            json.dump(self.radicalname2idx, f)
        
        self.writer2idx = writer2idx
        if self.writer2idx is not None:
            with open(os.path.join(self.save_path, "writer2idx.json"), "w") as f:
                json.dump(self.writer2idx, f)
        
        # model info
        
        self.image_size = image_size
        self.dim_char_embedding = dim_char_embedding
        self.char_length = char_length
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        
        self.learning_rate = learning_rate
        self.ema_beta = ema_beta
        self.diffusion_noise_steps = diffusion_noise_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        
        self.diversity_lambda = diversity_lambda
        
        with open(os.path.join(self.save_path, "model_info.json"), "w") as f:
            info = {
                "image_size": self.image_size,
                "dim_char_embedding": self.dim_char_embedding,
                "char_length": self.char_length,
                "num_res_blocks": self.num_res_blocks,
                "num_heads": self.num_heads,
                
                "learning_rate": self.learning_rate,
                "ema_beta": self.ema_beta,
                "diffusion_noise_steps": self.diffusion_noise_steps,
                "diffusion_beta_start": self.diffusion_beta_start,
                "diffusion_beta_end": self.diffusion_beta_end,
                
                "diversity_lambda": self.diversity_lambda,
            }
            json.dump(info, f)
        
        # create layers
        
        self.unet = UNetModel(
            image_size=self.image_size,
            in_channels=STABLE_DIFFUSION_CHANNEL,
            model_channels=self.dim_char_embedding,
            out_channels=STABLE_DIFFUSION_CHANNEL,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_classes=(len(self.writer2idx) if self.writer2idx is not None else None),
            num_heads=self.num_heads,
            context_dim=self.dim_char_embedding,
            vocab_size=len(self.radicalname2idx),
            char_length=char_length,
        ).to(device)
        
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        
        self.ema = EMA(self.ema_beta)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        
        self.diffusion = Diffusion(
            noise_steps=self.diffusion_noise_steps,
            beta_start=self.diffusion_beta_start,
            beta_end=self.diffusion_beta_end,
            image_size=self.image_size,
            device=self.device,
        )
    
    @staticmethod
    def load(
        save_path,
        stable_diffusion_path,
        
        device,
    ):
        print("loading RadicalStylist...")
        
        if not os.path.exists(save_path):
            raise Exception(f"not found: {save_path}")
            
        with open(os.path.join(save_path, "radicalname2idx.json")) as f:
            radicalname2idx = json.load(f)
        
        if os.path.isfile(os.path.join(save_path, "writer2idx.json")):
            with open(os.path.join(save_path, "writer2idx.json")) as f:
                writer2idx = json.load(f)
        else:
            writer2idx = None
        
        with open(os.path.join(save_path, "model_info.json")) as f:
            model_info = json.load(f)
            
            image_size = model_info["image_size"]
            dim_char_embedding = model_info["dim_char_embedding"]
            char_length = model_info["char_length"]
            num_res_blocks = model_info["num_res_blocks"]
            num_heads = model_info["num_heads"]
            
            learning_rate = model_info["learning_rate"]
            ema_beta = model_info["ema_beta"]
            diffusion_noise_steps = model_info["diffusion_noise_steps"]
            diffusion_beta_start = model_info["diffusion_beta_start"]
            diffusion_beta_end = model_info["diffusion_beta_end"]
            
            diversity_lambda = model_info["diversity_lambda"]
        
        instance = RadicalStylist(
            save_path,
            stable_diffusion_path,

            radicalname2idx,
            writer2idx,

            image_size,
            dim_char_embedding,
            char_length,
            num_res_blocks,
            num_heads,

            learning_rate,
            ema_beta,
            diffusion_noise_steps,
            diffusion_beta_start,
            diffusion_beta_end,
            
            diversity_lambda,

            device,
        )
        
        print("\tloading UNetModel...")
        
        instance.unet.load_state_dict(torch.load(os.path.join(save_path, "models", "unet.pt")))
        instance.unet.eval() # これいる？
        
        print("\tloaded.")
        
        print("\tloading optimizer...")
        
        instance.optimizer.load_state_dict(torch.load(os.path.join(save_path, "models", "optimizer.pt")))
        
        print("\tloaded.")
        
        print("\tloading EMA...")
        
        instance.ema_model.load_state_dict(torch.load(os.path.join(save_path, "models", "ema_model.pt")))
        instance.ema_model.eval() # これいる？
        
        print("\tloaded.")
        
        print("loaded.")
        
        return instance
    
    def train(self, data_loader: DataLoader[ConcatDataset], epochs, test_chars, test_writers):
        if os.path.exists(os.path.join(self.save_path, "train_info.json")):
            raise Exception("already trained")
        
        os.makedirs(os.path.join(self.save_path, "models"))
        os.makedirs(os.path.join(self.save_path, "generated"))
        
        num_epochs_digit = len(str(epochs))
        num_test_chars_digit = len(str(len(test_chars)))
        
        if epochs < 1000:
            checkpoint_epochs = set([0, epochs - 1])
        else:
            # ex) epochs = 1000 => checkpoint_epochs = {0, 99, 199, ..., 899, 999}
            tmp = 10 ** (num_epochs_digit - 2)
            checkpoint_epochs = set(i - 1 for i in range(tmp, epochs, tmp))
            checkpoint_epochs.add(0)
            checkpoint_epochs.add(epochs - 1)
            del tmp
        
        mse_loss = nn.MSELoss()
        
        loss_list = []

        self.unet.train()
        
        for epoch in range(epochs):
            loss_list.append(0)
            
            pbar = tqdm(data_loader, desc=f"{epoch=}")
            for i, (images, chars, writers) in enumerate(pbar):
                images = images.to(dtype=torch.float32, device=self.device)
                images = self.vae.encode(images).latent_dist.sample()
                images = images * 0.18215
                
                for char in chars:
                    random.shuffle(char.radicals)
                    for radical in char.radicals:
                        radical.idx = self.radicalname2idx[radical.name]
                
                if self.writer2idx is None:
                    writers_idx = None
                    
                else:
                    for i in range(len(writers)):
                        writers[i] = self.writer2idx[writers[i]]

                    writers_idx = torch.tensor(writers, dtype=torch.long, device=self.device)

                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                
                predicted_noise = self.unet(x_t, t, chars, writers_idx)
                
                loss = mse_loss(noise, predicted_noise)
                
                # # L_div
                # perm0 = np.random.permutation(images.shape[0])
                # perm1 = np.random.permutation(images.shape[0])
                # for i, j in zip(perm0, perm1):
                #     if i == j:
                #         continue
                #     loss -= self.diversity_lambda * nn.functional.l1_loss(predicted_noise[i], predicted_noise[j]) / nn.functional.l1_loss(images[i], images[j])
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.unet)
                
                loss = loss.item()
                loss_list[-1] += loss
                
                pbar.set_postfix(loss=loss)
            
            loss_list[-1] /= len(pbar)
            
            if epoch in checkpoint_epochs:
                images_list = self.sampling(test_chars, test_writers)
                for i, images in enumerate(images_list):
                    path = os.path.join(
                        self.save_path,
                        "generated",
                        f"test_{str(i).zfill(num_test_chars_digit)}_{str(epoch + 1).zfill(num_epochs_digit)}.jpg")
                    save_images(images, path)

                torch.save(self.unet.state_dict(), os.path.join(self.save_path, "models", "unet.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "models", "optimizer.pt"))
                torch.save(self.ema_model.state_dict(), os.path.join(self.save_path, "models", "ema_model.pt"))
                
                with open(os.path.join(self.save_path, "train_info.json"), "w") as f:
                    info = {
                        "data_loader": {
                            "batch_size": data_loader.batch_size,
                            "datasets": [d.rs_dataset_info for d in data_loader.dataset.datasets],
                        },
                        "epochs": epochs,
                        "loss": loss_list,
                        "test": {
                            "chars": [c.to_dict() for c in test_chars],
                            "writers": test_writers,
                        },
                    }
                    json.dump(info, f)

    def sampling(self, chars, writers):
        chars = copy.deepcopy(chars)
        for char in chars:
            for radical in char.radicals:
                radical.idx = self.radicalname2idx[radical.name]
        
        if type(writers) == int:
            n_per_chars = writers
            
            tmp_chars = []
            for c in chars:
                for _ in range(n_per_chars):
                    tmp_chars.append(c)
            chars = tmp_chars
            del tmp_chars
            
            writers_idx = writers
        
        else:
            n_per_chars = len(writers)
            
            writers_idx = [self.writer2idx[w] for w in writers]
            
            tmp_chars = []
            tmp_writers_idx = []
            for c in chars:
                for w in writers_idx:
                    tmp_chars.append(c)
                    tmp_writers_idx.append(w)
            chars, writers_idx = tmp_chars, tmp_writers_idx
            del tmp_chars, tmp_writers_idx
            
        ema_sampled_images = self.diffusion.sampling(
            self.ema_model, self.vae, chars, writers_idx
        )
        
        # char 毎にして返す
        ret = []
        for i in range(0, ema_sampled_images.shape[0], n_per_chars):
            ret.append(ema_sampled_images[i:(i + n_per_chars)])
        
        return ret
