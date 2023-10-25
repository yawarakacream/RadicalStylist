import copy
import json
import os
import sys

if not all(map(lambda p: p.endswith("stable_diffusion"), sys.path)):
    sys.path.append(os.path.join(os.path.dirname(__file__), "stable_diffusion"))

from tqdm import tqdm

import torch
from torch import optim, nn

from diffusion import EMA, Diffusion
from image_vae import StableDiffusionVae
from unet import UNetModel
from utility import save_images


class RadicalStylist:
    def __init__(
        self,
        
        save_path,
        stable_diffusion_path,
        
        radicalname2idx,
        writername2idx,
        
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
        
        self.vae = StableDiffusionVae(stable_diffusion_path, image_size, device)
        
        self.radicalname2idx = radicalname2idx
        with open(os.path.join(self.save_path, "radicalname2idx.json"), "w") as f:
            json.dump(self.radicalname2idx, f)
        
        self.writername2idx = writername2idx
        if self.writername2idx is not None:
            with open(os.path.join(self.save_path, "writername2idx.json"), "w") as f:
                json.dump(self.writername2idx, f)
        
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
            image_size=self.vae.image_size,
            in_channels=self.vae.num_channels,
            model_channels=self.dim_char_embedding,
            out_channels=self.vae.num_channels,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_classes=(len(self.writername2idx) if self.writername2idx is not None else None),
            num_heads=self.num_heads,
            context_dim=self.dim_char_embedding,
            vocab_size=len(self.radicalname2idx),
            char_length=char_length,
        ).to(device)
        
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        
        self.ema = EMA(self.ema_beta)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        
        self.diffusion = Diffusion(
            image_size=self.vae.image_size,
            num_image_channels=self.vae.num_channels,
            noise_steps=self.diffusion_noise_steps,
            beta_start=self.diffusion_beta_start,
            beta_end=self.diffusion_beta_end,
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
        
        if os.path.isfile(os.path.join(save_path, "writername2idx.json")):
            with open(os.path.join(save_path, "writername2idx.json")) as f:
                writername2idx = json.load(f)
        else:
            writername2idx = None
        
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
            writername2idx,

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
        instance.unet.eval()
        
        print("\tloaded.")
        
        print("\tloading optimizer...")
        
        instance.optimizer.load_state_dict(torch.load(os.path.join(save_path, "models", "optimizer.pt")))
        
        print("\tloaded.")
        
        print("\tloading EMA...")
        
        instance.ema_model.load_state_dict(torch.load(os.path.join(save_path, "models", "ema_model.pt")))
        instance.ema_model.eval()
        
        print("\tloaded.")
        
        print("loaded.")
        
        return instance
    
    def train(self, dataloader, epochs, test_chars, test_writers):
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
            
            pbar = tqdm(dataloader, desc=f"{epoch=}")
            for images, chars, writerindices in pbar:
                # prepare batch
                images = self.vae.encode(images)
                
                if self.writername2idx is None:
                    writerindices = None
                else:
                    writerindices = torch.tensor(writerindices, dtype=torch.long, device=self.device)

                # train
                ts = self.diffusion.sample_timesteps(images.shape[0])
                x_t, noise = self.diffusion.noise_images(images, ts)
                
                predicted_noise = self.unet(x_t, ts, chars, writerindices)
                
                loss = mse_loss(noise, predicted_noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.unet)
                
                loss = loss.item()
                loss_list[-1] += loss
                
                pbar.set_postfix(loss=loss)
            
            loss_list[-1] /= len(pbar)
            
            # checkpoint
            if epoch in checkpoint_epochs:
                images_list = self.sample(test_chars, test_writers)
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
                        "dataloader": {
                            "batch_size": dataloader.batch_size,
                            "dataset": dataloader.dataset.info,
                        },
                        "epochs": epochs,
                        "loss": loss_list,
                        "test": {
                            "chars": [c.to_dict() for c in test_chars],
                            "writers": test_writers,
                        },
                    }
                    json.dump(info, f)

    def sample(self, chars, writers):
        if isinstance(writers, int):
            assert 0 < writers
            
            n_per_chars = writers
            
            tmp_chars = []
            for c in chars:
                for _ in range(n_per_chars):
                    tmp_chars.append(c)
            
            chars = copy.deepcopy(tmp_chars)
            writerindices = None
        
        elif isinstance(writers, list):
            assert isinstance(writers[0], str)
            
            n_per_chars = len(writers)
            
            tmp_chars = []
            tmp_writerindices = []
            for c in chars:
                for w in writers:
                    tmp_chars.append(c)
                    tmp_writerindices.append(self.writername2idx[w])
                    
            chars = copy.deepcopy(tmp_chars)
            writerindices = torch.tensor(tmp_writerindices, dtype=torch.long, device=self.device)
            
        else:
            raise Exception()
            
        sampled = self.diffusion.sampling(self.ema_model, chars, writerindices)
        sampled = self.vae.decode(sampled)
        
        # char 毎にして返す
        ret = [sampled[i:(i + n_per_chars)] for i in range(0, sampled.shape[0], n_per_chars)]
        return ret
