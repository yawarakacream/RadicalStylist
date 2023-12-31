import copy
import json
import os
from typing import Optional, Union

from utility import pathstr, add_sys_path


add_sys_path(pathstr(os.path.dirname(__file__), "stable_diffusion"))


from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import DataloaderItem, RSDataset, Radical
from diffusion import EMA, Diffusion
from image_vae import StableDiffusionVae
from unet import UNetModel
from utility import pathstr, get_checkpoint_epochs, save_images


class RadicalStylist:
    def __init__(
        self,
        *,
        
        save_path: str,
        
        radicalname2idx: dict[str, int],
        writername2idx: Optional[dict[str, int]],

        vae: StableDiffusionVae,
        
        image_size: int,
        dim_radical_embedding: int,
        len_radicals_of_char: int,
        radical_position: str,
        num_res_blocks: int,
        num_heads: int,
        
        learning_rate: float,
        ema_beta: float,
        diffusion_noise_steps: int,
        diffusion_beta_start: float,
        diffusion_beta_end: float,
        
        device,
    ):
        self.device = device
        
        self.save_path = save_path
        
        self.radicalname2idx = radicalname2idx
        self.writername2idx = writername2idx
        
        self.vae = vae
        self.vae.to(device=self.device)
        
        self.image_size = image_size
        self.dim_radical_embedding = dim_radical_embedding
        self.len_radicals_of_char = len_radicals_of_char
        self.radical_position = radical_position
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        
        self.learning_rate = learning_rate
        self.ema_beta = ema_beta
        self.diffusion_noise_steps = diffusion_noise_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        
        # create layers
        
        self.unet = UNetModel(
            image_size=self.vae.calc_latent_size(self.image_size),
            in_channels=self.vae.latent_channels,
            model_channels=self.dim_radical_embedding,
            out_channels=self.vae.latent_channels,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_classes=(len(self.writername2idx) if self.writername2idx is not None else None),
            num_heads=self.num_heads,
            context_dim=self.dim_radical_embedding,
            vocab_size=len(self.radicalname2idx),
            len_radicals_of_char=len_radicals_of_char,
            radical_position=radical_position,
        ).to(device)
        
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        
        self.ema = EMA(self.ema_beta)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        
        self.diffusion = Diffusion(
            image_size=self.vae.calc_latent_size(self.image_size),
            num_image_channels=self.vae.latent_channels,
            noise_steps=self.diffusion_noise_steps,
            beta_start=self.diffusion_beta_start,
            beta_end=self.diffusion_beta_end,
            device=self.device,
        )
    
    @staticmethod
    def load(*, save_path: str, model_name: str, device: torch.device):
        if not os.path.exists(save_path):
            raise Exception(f"not found: {save_path}")

        with open(pathstr(save_path, "radicalname2idx.json")) as f:
            radicalname2idx = json.load(f)
        
        if os.path.isfile(pathstr(save_path, "writername2idx.json")):
            with open(pathstr(save_path, "writername2idx.json")) as f:
                writername2idx = json.load(f)
        else:
            writername2idx = None
        
        with open(pathstr(save_path, "model_info.json")) as f:
            model_info = json.load(f)

            vae_path = model_info["vae_path"]
            
            image_size = model_info["image_size"]
            dim_radical_embedding = model_info["dim_radical_embedding"]
            len_radicals_of_char = model_info["len_radicals_of_char"]
            radical_position = model_info["radical_position"]
            num_res_blocks = model_info["num_res_blocks"]
            num_heads = model_info["num_heads"]
            
            learning_rate = model_info["learning_rate"]
            ema_beta = model_info["ema_beta"]
            diffusion_noise_steps = model_info["diffusion_noise_steps"]
            diffusion_beta_start = model_info["diffusion_beta_start"]
            diffusion_beta_end = model_info["diffusion_beta_end"]
        
        vae = StableDiffusionVae(vae_path)

        instance = RadicalStylist(
            save_path=save_path,

            radicalname2idx=radicalname2idx,
            writername2idx=writername2idx,

            vae=vae,

            image_size=image_size,
            dim_radical_embedding=dim_radical_embedding,
            len_radicals_of_char=len_radicals_of_char,
            radical_position=radical_position,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,

            learning_rate=learning_rate,
            ema_beta=ema_beta,
            diffusion_noise_steps=diffusion_noise_steps,
            diffusion_beta_start=diffusion_beta_start,
            diffusion_beta_end=diffusion_beta_end,

            device=device,
        )
        
        instance.unet.load_state_dict(torch.load(pathstr(save_path, "models", model_name, "unet.pt"), map_location=device))
        instance.unet.eval()

        instance.optimizer.load_state_dict(torch.load(pathstr(save_path, "models", model_name, "optimizer.pt"), map_location=device))

        instance.ema_model.load_state_dict(torch.load(pathstr(save_path, "models", model_name, "ema_model.pt"), map_location=device))
        instance.ema_model.eval()
        
        return instance
    
    def save(self, *, model_name: str, exist_ok=True):
        if (not exist_ok) and os.path.exists(self.save_path):
            raise Exception(f"already exists: {self.save_path}")

        os.makedirs(self.save_path, exist_ok=True)

        with open(pathstr(self.save_path, "radicalname2idx.json"), "w") as f:
            json.dump(self.radicalname2idx, f)

        if self.writername2idx is not None:
            with open(pathstr(self.save_path, "writername2idx.json"), "w") as f:
                json.dump(self.writername2idx, f)

        with open(pathstr(self.save_path, "model_info.json"), "w") as f:
            info = {
                "vae_path": self.vae.vae_path,

                "image_size": self.image_size,
                "dim_radical_embedding": self.dim_radical_embedding,
                "len_radicals_of_char": self.len_radicals_of_char,
                "radical_position": self.radical_position,
                "num_res_blocks": self.num_res_blocks,
                "num_heads": self.num_heads,
                
                "learning_rate": self.learning_rate,
                "ema_beta": self.ema_beta,
                "diffusion_noise_steps": self.diffusion_noise_steps,
                "diffusion_beta_start": self.diffusion_beta_start,
                "diffusion_beta_end": self.diffusion_beta_end,
            }
            json.dump(info, f)

        os.makedirs(pathstr(self.save_path, "models", model_name))

        torch.save(self.unet.state_dict(), pathstr(self.save_path, "models", model_name, "unet.pt"))
        torch.save(self.optimizer.state_dict(), pathstr(self.save_path, "models", model_name, "optimizer.pt"))
        torch.save(self.ema_model.state_dict(), pathstr(self.save_path, "models", model_name, "ema_model.pt"))

        os.makedirs(pathstr(self.save_path, "generated"), exist_ok=True)

    def train(
        self,
        train_dataloader: DataLoader[DataloaderItem],
        epochs: int,
        test_radicallists_with_name: list[tuple[str, list[Radical]]],
        test_writers: Union[list[str], int],
    ) -> None:
        if os.path.exists(pathstr(self.save_path, "train_info.json")):
            raise Exception("already trained")
        
        if not isinstance(train_dataloader.dataset, RSDataset):
            raise Exception(f"illegal train dataset: {train_dataloader.dataset}")

        num_epochs_digit = len(str(epochs))
        num_test_radicallists_digit = len(str(len(test_radicallists_with_name)))
        
        if 500000 <= len(train_dataloader.dataset):
            checkpoint_epochs = get_checkpoint_epochs(epochs, step=10)
        elif 100000 <= len(train_dataloader.dataset):
            checkpoint_epochs = get_checkpoint_epochs(epochs, step=50)
        else:
            checkpoint_epochs = get_checkpoint_epochs(epochs, step=100)
        
        test_radicallists = [r for _, r in test_radicallists_with_name]
        
        loss_list = []

        self.unet.train()
        
        for epoch in range(epochs):
            loss_list.append(0)
            
            pbar: tqdm[DataloaderItem] = tqdm(train_dataloader, desc=f"{epoch=}")
            for images, radicallists, writerindices in pbar:
                # prepare batch
                images = self.vae.encode(images)

                if self.writername2idx is None:
                    assert writerindices is None
                else:
                    writerindices = torch.tensor(writerindices, dtype=torch.long, device=self.device)

                # train
                ts = self.diffusion.sample_timesteps(images.shape[0])
                x_t, noise = self.diffusion.noise_images(images, ts)

                predicted_noise = self.unet(x_t, ts, radicallists, writerindices)

                loss = F.mse_loss(noise, predicted_noise, reduction="mean")
                
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
                checkpoint_name = str(epoch + 1).zfill(num_epochs_digit)
                print(f"checkpoint {checkpoint_name}:")

                images_list = self.sample(test_radicallists, test_writers)
                for i, images in enumerate(images_list):
                    path = pathstr(
                        self.save_path,
                        "generated",
                        f"test_{str(i).zfill(num_test_radicallists_digit)}_{checkpoint_name}.png",
                    )
                    save_images(images, path)
                
                with open(pathstr(self.save_path, "train_info.json"), "w") as f:
                    train_info = {
                        "dataloader": {
                            "batch_size": train_dataloader.batch_size,
                            "dataset": train_dataloader.dataset.info,
                        },
                        "epochs": epochs,
                        "loss": loss_list,
                        "test": {
                            "radicallists": [
                                {"name": name, "elements": [r.to_dict() for r in radicallist]}
                                for name, radicallist in test_radicallists_with_name
                            ],
                            "writers": test_writers,
                        },
                    }
                    json.dump(train_info, f)

                self.save(model_name=f"checkpoint={checkpoint_name}")

    @torch.no_grad()
    def sample(self, radicallists, writers):
        if isinstance(writers, int):
            assert self.writername2idx is None
            assert 0 < writers
            
            n_per_chars = writers
            
            tmp_radicallists = []
            for radicallist in radicallists:
                for _ in range(n_per_chars):
                    tmp_radicallists.append(radicallist)

            radicallists = copy.deepcopy(tmp_radicallists)
            writerindices = None
        
        elif isinstance(writers, list):
            assert self.writername2idx is not None
            assert all(map(lambda w: isinstance(w, str), writers))
            
            n_per_chars = len(writers)
            
            tmp_radicallists = []
            tmp_writerindices = []
            for radicallist in radicallists:
                for writer in writers:
                    tmp_radicallists.append(radicallist)
                    tmp_writerindices.append(self.writername2idx[writer])
                    
            radicallists = copy.deepcopy(tmp_radicallists)
            writerindices = torch.tensor(tmp_writerindices, dtype=torch.long, device=self.device)
            
        else:
            raise Exception()

        sampled_latents = self.diffusion.sample(self.ema_model, radicallists, writerindices)
        sampled_images = self.vae.decode(sampled_latents)
        
        # char 毎にして返す
        ret = [
            sampled_images[i:(i + n_per_chars)]
            for i in range(0, sampled_images.size(0), n_per_chars)
        ]
        return ret
