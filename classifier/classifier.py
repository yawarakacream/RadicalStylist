import json
import os

from tqdm import tqdm

import numpy as np

import torch
from torch import optim, nn

from utility import pathstr, rgb_to_grayscale
from image_vae import StableDiffusionVae


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class RadicalClassifier(nn.Module):
    def __init__(self, image_size: int, image_channels: int, num_radicals: int):
        super(RadicalClassifier, self).__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_radicals = num_radicals

        self.hidden0 = nn.Sequential(
            nn.Conv2d(self.image_channels, 16, kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(num_radicals)
        ])
        
    def forward(self, latents):
        x = latents
        x = self.hidden0(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.hidden1(x)
        x = [self.out[i](x) for i in range(self.num_radicals)]
        x = torch.cat(x, dim=1) # [batch_size, num of radical]
        return x


class RSClassifier:
    def __init__(
        self,
        
        save_path: str,
        
        radicalname2idx: dict[str, int],

        vae: StableDiffusionVae,
        
        image_size: int,
        grayscale: bool,
        
        learning_rate: float,
        bce_pos_weight: list[float],
        
        device: torch.device,
    ):
        self.device = device
        
        self.save_path = save_path
        
        self.vae = vae
        
        self.radicalname2idx = radicalname2idx
        self.radicalidx2name = {idx: name for name, idx in radicalname2idx.items()}
        
        self.image_size = image_size
        self.grayscale = grayscale
        
        self.learning_rate = learning_rate
        self.bce_pos_weight = bce_pos_weight
        
        self.classifier = RadicalClassifier(
            image_size=self.vae.calc_latent_size(self.image_size),
            image_channels=self.vae.latent_channels,
            num_radicals=len(radicalname2idx),
        ).to(device=device)
        self.classifier.requires_grad_(True)
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=learning_rate)

    @staticmethod
    def load(save_path: str, vae: StableDiffusionVae, device: torch.device):
        if not os.path.exists(save_path):
            raise Exception(f"not found: {save_path}")
        
        with open(pathstr(save_path, "radicalname2idx.json")) as f:
            radicalname2idx = json.load(f)
        
        with open(pathstr(save_path, "model_info.json")) as f:
            model_info = json.load(f)
            
            image_size = model_info["image_size"]
            grayscale = model_info["grayscale"]
            
            learning_rate = model_info["learning_rate"]
            bce_pos_weight = model_info["bce_pos_weight"]
        
        instance = RSClassifier(
            save_path=save_path,

            radicalname2idx=radicalname2idx,

            vae=vae,

            image_size=image_size,
            grayscale=grayscale,

            learning_rate=learning_rate,
            bce_pos_weight=bce_pos_weight,

            device=device,
        )
        
        instance.classifier.load_state_dict(torch.load(pathstr(save_path, "models", "classifier.pt")))
        instance.classifier.eval()
        
        instance.optimizer.load_state_dict(torch.load(pathstr(save_path, "models", "optimizer.pt")))

        return instance

    def save(self, *, exist_ok=True):
        if (not exist_ok) and os.path.exists(self.save_path):
            raise Exception(f"already exists: {self.save_path}")

        os.makedirs(self.save_path, exist_ok=True)
        
        with open(pathstr(self.save_path, "radicalname2idx.json"), "w") as f:
            json.dump(self.radicalname2idx, f)
        
        with open(pathstr(self.save_path, "model_info.json"), "w") as f:
            model_info = {
                "image_size": self.image_size,
                "grayscale": self.grayscale,
                
                "learning_rate": self.learning_rate,
                "bce_pos_weight": self.bce_pos_weight,
            }
            json.dump(model_info, f)
        
        os.makedirs(pathstr(self.save_path, "models"), exist_ok=True)
        
        torch.save(self.classifier.state_dict(), pathstr(self.save_path, "models", "classifier.pt"))
        torch.save(self.optimizer.state_dict(), pathstr(self.save_path, "models", "optimizer.pt"))

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
    
    def predict_radicals(self, latents, one_hot=False):
        predictions = self.classifier(latents)
        predictions = torch.sigmoid(predictions)
        if one_hot:
            predictions = torch.where(predictions > 0.5, 1, 0)
        return predictions

    def train(
        self,

        train_dataloader,
        valid_dataloader,

        epochs: int,
    ):
        self.classifier.train()
        
        criterion = nn.BCEWithLogitsLoss(
            # 大きくすれば recall が，小さくすれば precision が上がる
            pos_weight=torch.tensor(self.bce_pos_weight, dtype=torch.float),
        ).to(device=self.device)
        
        # log
        loss_list = np.zeros(epochs)

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
                # prepare batch
                if self.grayscale:
                    images = rgb_to_grayscale(images)

                latents = self.vae.encode(images)

                # train
                radical_answers = self.prepare_answers(chars, dtype=torch.float)
                radical_predictions = self.classifier(latents)
                
                loss = criterion(radical_predictions, radical_answers)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                loss_list[epoch] += loss
                pbar.set_postfix(loss=loss)

            if epoch in checkpoint_epochs:
                for dataname, dataloader in (
                    ("train", train_dataloader),
                    ("valid", valid_dataloader),
                ):
                    ap_count_0 = self.evaluate(dataloader)
                    ap_count[dataname][epoch] += ap_count_0

                with open(pathstr(self.save_path, "train_info.json"), "w") as f:
                    info = {
                        "epochs": epochs,
                    }
                    json.dump(info, f)

                np.save(pathstr(self.save_path, "loss_list.npy"), loss_list, allow_pickle=True)
                np.save(pathstr(self.save_path, "ap_count.npy"), ap_count, allow_pickle=True) # type: ignore

                self.save()

    @torch.no_grad()
    def evaluate(self, dataloader):
        ap_count = np.zeros((2, 2), dtype=np.int64)

        for i, (images, chars, _) in enumerate(tqdm(dataloader)):
            if self.grayscale:
                images = rgb_to_grayscale(images)
                
            latents = self.vae.encode(images)
            
            # ap_count
            radical_answers = torch.flatten(self.prepare_answers(chars))
            radical_predictions = torch.flatten(self.predict_radicals(latents, one_hot=True))
            for a, p in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ap_count[a, p] += torch.count_nonzero(
                    torch.logical_and(radical_answers == a, radical_predictions == p).long()
                )

        return ap_count
