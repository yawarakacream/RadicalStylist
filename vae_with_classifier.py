import json
import os
import random
import sys

from tqdm import tqdm

import numpy as np

import torch
from torch import optim, nn
from torch.nn import functional as F

import torchvision

from diffusers import AutoencoderKL

# from stable_diffusion.ldm.modules.losses import LPIPSWithDiscriminator

import lpips

from dataset import RSDataset
from utility import pathstr, save_images, rgb_to_grayscale
    

class RadicalClassifier(nn.Module):
    def __init__(self, image_size, image_channels, num_radicals):
        super(RadicalClassifier, self).__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_radicals = num_radicals
        
        self.hidden = nn.Sequential(
            nn.Linear((self.image_size ** 2) * self.image_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            for _ in range(num_radicals)
        ])
        
    def forward(self, latents):
        x = torch.flatten(latents, 1) # flatten all dimensions except batch
        x = self.hidden(x)
        x = [self.out[i](x) for i in range(self.num_radicals)]
        x = torch.cat(x, dim=1) # [batch_size, num of radical]
        return x


class VaeWithClassifier:
    def __init__(
        self,
        
        save_path,
        
        vae,
        classifier,
        optimizer,
        
        charname2radicaljson,
        radicalname2idx,
        
        image_size,
        grayscale,
        
        vae_loss_kl_weight,
        vae_loss_lpips_weight,
        cf_loss_weight,
        cf_loss_bce_pos_weight,
        
        device,
    ):
        self.device = device
        
        self.save_path = save_path
        
        self.vae = vae
        self.classifier = classifier
        self.optimizer = optimizer
        
        self.charname2radicaljson = charname2radicaljson
        self.radicalname2idx = radicalname2idx
        
        self.image_size = image_size
        self.grayscale = grayscale
        
        self.vae_loss_kl_weight = vae_loss_kl_weight
        self.vae_loss_lpips_weight = vae_loss_lpips_weight
        self.cf_loss_weight = cf_loss_weight
        self.cf_loss_bce_pos_weight = cf_loss_bce_pos_weight
        
        self.normalize_image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.save()
    
    def new(
        save_path,
        stable_diffusion_path,
        
        charname2radicaljson,
        radicalname2idx,
        
        image_size,
        grayscale,
        
        learning_rate,
        vae_loss_kl_weight,
        vae_loss_lpips_weight,
        cf_loss_weight,
        cf_loss_bce_pos_weight,
        
        device,
    ):
        if os.path.exists(save_path):
            raise Exception(f"already exists: {save_path}")
        
        vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)
        classifier = RadicalClassifier(image_size // 8, 4, len(radicalname2idx)).to(device=device)
        optimizer = optim.AdamW(
            [{"params": vae.parameters()}, {"params": classifier.parameters()}],
            lr=learning_rate,
        )
        
        return VaeWithClassifier(
            save_path=save_path,

            vae=vae,
            classifier=classifier,
            optimizer=optimizer,

            charname2radicaljson=charname2radicaljson,
            radicalname2idx=radicalname2idx,

            image_size=image_size,
            grayscale=grayscale,
            
            vae_loss_kl_weight=vae_loss_kl_weight,
            vae_loss_lpips_weight=vae_loss_lpips_weight,
            cf_loss_weight=cf_loss_weight,
            cf_loss_bce_pos_weight=cf_loss_bce_pos_weight,

            device=device,
        )
    
    def load(save_path, charname2radicaljson, device):
        with open(os.path.join(save_path, "radicalname2idx.json")) as f:
            radicalname2idx = json.load(f)
            
        with open(os.path.join(save_path, "model_info.json")) as f:
            model_info = json.load(f)
            
            image_size = model_info["image_size"]
            grayscale = model_info["grayscale"]
            
            vae_loss_kl_weight = model_info["vae_loss_kl_weight"]
            vae_loss_lpips_weight = model_info["vae_loss_lpips_weight"]
            cf_loss_weight = model_info["cf_loss_weight"]
            cf_loss_bce_pos_weight = model_info["cf_loss_bce_pos_weight"]
            
        vae = AutoencoderKL.from_pretrained(pathstr(self.save_path, "models", "vae")).to(device)
        
        classifier = RadicalClassifier(image_size // 8, 4, len(radicalname2idx)).to(device=device)
        classifier.load_state_dict(torch.load(pathstr(save_path, "models", "classifier.pt")))
        classifier.eval()
        
        optimizer = optim.AdamW(
            [{"params": vae.parameters()}, {"params": classifier.parameters()}],
            lr=learning_rate,
        )
        optimizer.load_state_dict(torch.load(pathstr(save_path, "models", "optimizer.pt")))
        
        return VaeWithClassifier(
            save_path=save_path,

            vae=vae,
            classifier=classifier,
            optimizer=optimizer,

            charname2radicaljson=charname2radicaljson,
            radicalname2idx=radicalname2idx,

            image_size=image_size,
            grayscale=grayscale,
            
            vae_loss_kl_weight=vae_loss_kl_weight,
            vae_loss_lpips_weight=vae_loss_lpips_weight,
            cf_loss_weight=cf_loss_weight,
            cf_loss_bce_pos_weight=cf_loss_bce_pos_weight,

            device=device,
        )
        
    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        
        with open(pathstr(self.save_path, "radicalname2idx.json"), "w") as f:
            json.dump(self.radicalname2idx, f)
        
        with open(pathstr(self.save_path, "model_info.json"), "w") as f:
            info = {
                "image_size": self.image_size,
                "grayscale": self.grayscale,
                
                "vae_loss_kl_weight": self.vae_loss_kl_weight,
                "vae_loss_lpips_weight": self.vae_loss_lpips_weight,
                "cf_loss_weight": self.cf_loss_weight,
                "cf_loss_bce_pos_weight": self.cf_loss_bce_pos_weight,
            }
            json.dump(info, f)
        
        self.vae.save_pretrained(pathstr(self.save_path, "models", "vae"))
        
        torch.save(self.classifier.state_dict(), pathstr(self.save_path, "models", "classifier.pt"))
        torch.save(self.optimizer.state_dict(), pathstr(self.save_path, "models", "optimizer.pt"))

    @property
    def latent_channels(self):
        return self.vae.latent_channels
    
    @property
    def latent_size(self):
        return self.image_size // 8
        
    def prepare_answers(self, chars, dtype=torch.long):
        answers = [] # batch_size, radicalidx
        for char in chars:
            answers.append([])
            for radicalidx in self.radicalname2idx.values():
                answers[-1].append(0)
                for radical in char.radicals:
                    if radical.idx == radicalidx:
                        answers[-1][-1] = 1
        answers = torch.tensor(answers, dtype=dtype, device=self.device)
        return answers
    
    def encode(self, images):
        images = self.normalize_image(images)
        images = images.to(dtype=torch.float32, device=self.device)
        if self.grayscale:
            images = rgb_to_grayscale(images)

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        
        return latents
    
    def decode(self, latents):
        latents = 1 / 0.18215 * latents
        
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1) # 謎
        if self.grayscale:
            images = rgb_to_grayscale(images)
            
        return images
    
    def predict_radicals(self, latents):
        predictions = self.classifier(latents)
        predictions = torch.sigmoid(predictions)
        predictions = torch.where(predictions > 0.5, 1, 0)
        return predictions

    # https://github.com/huggingface/diffusers/issues/3726
    # https://github.com/huggingface/diffusers/pull/3801
    # https://github.com/runwayml/stable-diffusion/blob/08ab4d326c96854026c4eb3454cd3b02109ee982/ldm/modules/losses/contperceptual.py#L45
    def train(
        self,

        train_dataloader,
        valid_dataloader,

        epochs,
    ):
        self.vae.requires_grad_(True)
        self.classifier.requires_grad_(True)
        self.classifier.train()
        
        ignore_vae_train = True
        if ignore_vae_train:
            self.vae.requires_grad_(False)
        
        # classifier loss fn
        bce_loss_fn = nn.BCEWithLogitsLoss(
            # 大きくすれば recall が，小さくすれば precision が上がる
            pos_weight=torch.tensor(self.cf_loss_bce_pos_weight, dtype=torch.float),
        ).to(device=self.device)
        
        # vae loss fn
        mse_loss_fn = nn.MSELoss(reduction="mean")
        lpips_loss_fn = lpips.LPIPS(net="alex").to(self.device)

        # log
        vae_loss_list = np.zeros(epochs)
        cf_loss_list = np.zeros(epochs)

        ap_count = {
            "train": np.zeros((epochs, 2, 2), dtype=np.int64),
            "valid": np.zeros((epochs, 2, 2), dtype=np.int64),
        }

        num_epochs_digit = len(str(epochs))
        
        if epochs < 10:
            checkpoint_epochs = set([0, epochs - 1])
        else:
            # ex) epochs = 100 => checkpoint_epochs = {0, 4, 9, ..., 94, 99}
            tmp = min(10, (10 ** (num_epochs_digit - 2)) // 2)
            checkpoint_epochs = set(i - 1 for i in range(tmp, epochs, tmp))
            checkpoint_epochs.add(0)
            checkpoint_epochs.add(epochs - 1)
            del tmp
            
        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc=f"{epoch=}")
            for images, chars, _ in pbar:
                images = self.normalize_image(images)
                images = images.to(dtype=torch.float32, device=self.device)
                if self.grayscale:
                    images = rgb_to_grayscale(images)

                # vae encode
                posteriors = self.vae.encode(images).latent_dist
                latents = posteriors.mode()
                latents = latents * 0.18215

                # classifier loss
                radical_answers = self.prepare_answers(chars, dtype=torch.float)
                radical_predictions = self.classifier(latents)
                cf_loss = bce_loss_fn(radical_predictions, radical_answers)

                # vae decode
                latents = 1 / 0.18215 * latents
                reconstructed_images = self.vae.decode(latents).sample
                if self.grayscale:
                    reconstructed_images = rgb_to_grayscale(reconstructed_images)

                # vae loss
                kl_loss = posteriors.kl().mean()
                mse_loss = mse_loss_fn(reconstructed_images, images)
                lpips_loss = lpips_loss_fn(reconstructed_images, images).mean()
                vae_loss = mse_loss + kl_loss * self.vae_loss_kl_weight + lpips_loss * self.vae_loss_lpips_weight

                # sum loss
                if ignore_vae_train:
                    loss = cf_loss
                else:
                    loss = vae_loss + cf_loss * self.cf_loss_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                vae_loss = vae_loss.item()
                cf_loss = cf_loss.item()
                vae_loss_list[epoch] += vae_loss
                cf_loss_list[epoch] += cf_loss
                pbar.set_postfix(vae_loss=vae_loss, cf_loss=cf_loss)

            if epoch in checkpoint_epochs:
                self.save()
                
                os.makedirs(pathstr(self.save_path, "reconstruction"), exist_ok=True)
                
                for dataname, dataloader in (
                    ("train", train_dataloader),
                    ("valid", valid_dataloader),
                ):
                    (original_images_0, reconstructed_images_0), ap_count_0 = self.evaluate(dataloader)
                    
                    save_images(
                        original_images_0,
                        pathstr(self.save_path, "reconstruction", f"{dataname}_{str(epoch + 1).zfill(num_epochs_digit)}_before.jpg")
                    )
                    save_images(
                        reconstructed_images_0,
                        pathstr(self.save_path, "reconstruction", f"{dataname}_{str(epoch + 1).zfill(num_epochs_digit)}_after.jpg")
                    )
                    
                    ap_count[dataname][epoch] += ap_count_0
                
                with open(pathstr(self.save_path, "train_info.json"), "w") as f:
                    info = {
                        "epochs": epochs,
                    }
                    json.dump(info, f)

                np.save(pathstr(self.save_path, "vae_loss_list.npy"), vae_loss_list, allow_pickle=True)
                np.save(pathstr(self.save_path, "cf_loss_list.npy"), cf_loss_list, allow_pickle=True)
                np.save(pathstr(self.save_path, "ap_count.npy"), ap_count, allow_pickle=True)

    @torch.no_grad()
    def evaluate(self, dataloader):
        ap_count = np.zeros((2, 2), dtype=np.int64)

        for i, (images, chars, _) in enumerate(tqdm(dataloader)):
            latents = self.encode(images)
            
            # encode & decode
            if i == 0:
                original_images = images.detach()
                reconstructed_images = self.decode(latents)
                
            # ap_count
            radical_answers = torch.flatten(self.prepare_answers(chars))
            radical_predictions = torch.flatten(self.predict_radicals(latents))
            for a, p in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ap_count[a, p] += torch.count_nonzero(
                    torch.logical_and(radical_answers == a, radical_predictions == p).long()
                )

        return (original_images, reconstructed_images), ap_count
