import argparse

import torch

from classifier.classifier import RSClassifier
from image_vae import StableDiffusionVae
from utility import pathstr, read_image_as_tensor


def main(
    save_path: str,
    stable_diffusion_path: str,

    image_paths: list[str],
    
    device: torch.device,
) -> None:
    print(f"save_path: {save_path}")
    
    print("loading StableDiffusionVae...")
    vae = StableDiffusionVae(stable_diffusion_path, device)
    print("loaded.")

    print("loading RSClassifier...")
    rscf = RSClassifier.load(save_path=save_path, vae=vae, device=device)
    print("loaded.")

    radicalidx2name = {idx: name for name, idx in rscf.radicalname2idx.items()}

    images = torch.stack([read_image_as_tensor(p) for p in image_paths])
    latents = vae.encode(images)

    probabilities = rscf.predict_radicals(latents)

    for probs, image_path in zip(probabilities, image_paths):
        print(image_path)

        ranks = [(p, i) for i, p in enumerate(probs) if 0.5 < p.item()]
        ranks.sort(reverse=True)

        for p, idx in ranks:
            print(f"{radicalidx2name[idx]} (p = {p})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--image_paths", type=str, nargs="*")
    args = parser.parse_args()
    
    main(
        save_path=pathstr("./output/classifier ETL8G epochs=250"),
        stable_diffusion_path=pathstr("~/datadisk/stable-diffusion-v1-5"),

        image_paths=args.image_paths,
        
        device=torch.device(args.device),
    )
