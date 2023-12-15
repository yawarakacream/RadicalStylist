import json
import os
from typing import Any, Final

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import functional as TVF

from PIL import Image

from diffusers import AutoencoderKL

from dataset import DataloaderItem, RSDataset
from utility import pathstr, get_checkpoint_epochs, save_images


class StableDiffusionVae:
    vae_path: Final[str]
    vae: Final[Any]

    transforms: Final[nn.Module]

    latent_channels: Final[int] = 4

    def __init__(self, vae_path):
        if not os.path.exists(vae_path):
            # 存在しないパスを AutoencoderKL.from_pretrained に渡すと
            # huggingface に取りに行ってしまうので前もって確認する
            raise Exception(f"vae not found: {vae_path}")
        
        self.vae_path = vae_path
        self.vae = AutoencoderKL.from_pretrained(self.vae_path) # type: ignore
        assert isinstance(self.vae, AutoencoderKL)
        self.vae.requires_grad_(False)
                
        self.transforms = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @property
    def device(self) -> torch.device:
        return self.vae.device

    def to(self, *, device: torch.device):
        self.vae.to(device=device)
        return self

    @staticmethod
    def calc_latent_size(image_size: int):
        if image_size % 8 != 0:
            raise Exception("image_size must be a multiple of 8")
        return image_size // 8
        
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = self.transforms(images)
        images = images.to(dtype=torch.float32, device=self.device)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) # 謎
        return image

    def train(
        self,
        save_path: str,
        dataloader: DataLoader[DataloaderItem],
        epochs: int,
        test_imagepaths: list[str],
    ) -> None:
        self.vae.requires_grad_(True)

        if os.path.exists(save_path):
            raise Exception(f"already exists: {save_path}")
        os.makedirs(save_path, exist_ok=False)
        os.makedirs(pathstr(save_path, "reconstruction"), exist_ok=True)

        if not isinstance(dataloader.dataset, RSDataset):
            raise Exception(f"illegal dataset: {dataloader.dataset}")

        num_epochs_digit = len(str(epochs))

        checkpoint_epochs = get_checkpoint_epochs(epochs, step=10)

        test_images = []
        for p in test_imagepaths:
            with Image.open(p) as image:
                test_images.append(TVF.to_tensor(image.convert("RGB")))
        test_images = torch.stack(test_images).to(device=self.device)
        save_images(
            test_images,
            pathstr(save_path, "reconstruction", f"test_original.png"),
        )
        save_images(
            self.decode(self.encode(test_images)),
            pathstr(save_path, "reconstruction", f"test_before.png"),
        )
        
        self.optimizer = optim.AdamW(self.vae.parameters(), lr=0.0001)

        loss_list: list[float] = []
        test_loss_list: list[float] = []

        for epoch in range(epochs):
            loss_list.append(0)
            
            pbar: tqdm[DataloaderItem] = tqdm(dataloader, desc=f"{epoch=}")
            for images, _, _ in pbar:
                images = images.to(device=self.device)
                reconstructions = self.decode(self.encode(images))

                loss = F.mse_loss(images, reconstructions, reduction="mean")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                loss_list[-1] += loss
                pbar.set_postfix(loss=loss)

            loss_list[-1] /= len(pbar)

            with torch.no_grad():
                test_reconstructions = self.decode(self.encode(test_images))

                test_loss = F.mse_loss(test_images, test_reconstructions, reduction="mean")
                test_loss = test_loss.item()
                test_loss_list.append(test_loss)
                
            if epoch in checkpoint_epochs:
                checkpoint_name = str(epoch + 1).zfill(num_epochs_digit)
                print(f"checkpoint {checkpoint_name}:")

                save_images(
                    test_reconstructions,
                    pathstr(save_path, "reconstruction", f"test_{checkpoint_name}.png"),
                )

                plt.title("loss")
                plt.plot(range(epoch + 1), loss_list, label="train")
                plt.plot(range(epoch + 1), test_loss_list, label="test")
                plt.legend()
                plt.savefig(pathstr(save_path, "loss.png"))
                plt.close()

                print(f"test loss: {test_loss_list[-1]}")

                with open(pathstr(save_path, "train_info.json"), "w") as f:
                    train_info = {
                        "dataloader": {
                            "batch_size": dataloader.batch_size,
                            "dataset": dataloader.dataset.info,
                        },
                        "epochs": epochs,
                        "loss": {
                            "train": loss_list,
                            "test": test_loss_list,
                        },
                        "test": test_imagepaths,
                    }
                    json.dump(train_info, f)

                self.vae.save_pretrained(pathstr(save_path, f"vae_{checkpoint_name}"))
