import argparse
import os
import torch
import matplotlib.pyplot as plt

from train_pl import VAE
from utils import visualize_manifold


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = VAE.load_from_checkpoint(args.checkpoint)
    if model.hparams.z_dim != 2:
        raise ValueError(
            f"Checkpoint was trained with z_dim={model.hparams.z_dim}. "
            "Manifold visualization requires z_dim=2. Please provide a 2D-latent checkpoint."
        )
    model.eval().to(device)

    with torch.no_grad():
        grid = visualize_manifold(model.decoder, grid_size=args.grid_size)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    img = grid.permute(1, 2, 0).cpu().numpy()
    plt.imsave(args.out_path, img, cmap="gray")
    print(f"Saved manifold to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VAE manifold from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="VAE_logs/lightning_logs/version_2/checkpoints/epoch=76-step=32494.ckpt",
        help="Path to Lightning checkpoint",
    )
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size (per axis)")
    parser.add_argument(
        "--out_path",
        type=str,
        default="generated_samples/manifold_latent2.png",
        help="Where to save the manifold image",
    )
    main(parser.parse_args())
