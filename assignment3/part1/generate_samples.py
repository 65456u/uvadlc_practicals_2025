################################################################################
# Script to generate samples at different training stages
################################################################################

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import pytorch_lightning as pl

from train_pl import VAE
from mnist import mnist


def generate_and_save_samples(model, epoch_name, save_path, num_samples=64):
    """
    Generate samples from the model and save as 8x8 grid.
    
    Args:
        model: VAE model
        epoch_name: Name for the epoch (e.g., "epoch_0", "epoch_10", "epoch_80")
        save_path: Directory to save the image
        num_samples: Number of samples to generate (default 64 for 8x8 grid)
    """
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
        # Convert 4-bit images to [0, 1] range
        samples = samples.float() / 15.0
        
        # Create 8x8 grid
        grid = make_grid(samples, nrow=8, normalize=True, value_range=(0, 1), pad_value=0.5)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        
        # Plot and save
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray' if grid.shape[2] == 1 else None)
        plt.axis('off')
        plt.title(f'Generated Samples - {epoch_name}', fontsize=16)
        plt.tight_layout()
        
        output_file = os.path.join(save_path, f'{epoch_name}_samples.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved samples to: {output_file}")


def main():
    # Setup
    save_dir = "generated_samples"
    os.makedirs(save_dir, exist_ok=True)
    
    # Hyperparameters (should match your training config)
    z_dim = 20
    num_filters = 32
    lr = 1e-3
    batch_size = 128
    seed = 42
    
    pl.seed_everything(seed)
    
    # ===== 1. Generate samples from UNTRAINED model (Epoch 0) =====
    print("\n" + "="*60)
    print("Generating samples from UNTRAINED model (Epoch 0)...")
    print("="*60)
    model_epoch0 = VAE(num_filters=num_filters, z_dim=z_dim, lr=lr)
    generate_and_save_samples(model_epoch0, "epoch_0_untrained", save_dir)
    
    
    # ===== 2. Train progressively and generate samples at different stages =====
    # Load data
    train_loader, val_loader, _ = mnist(batch_size=batch_size, num_workers=0, root='../data/')
    
    # Train and sample at different epochs
    epochs_to_sample = [1, 5, 10, 20, 80]
    model = VAE(num_filters=num_filters, z_dim=z_dim, lr=lr)
    
    for target_epoch in epochs_to_sample:
        print("\n" + "="*60)
        print(f"Training model to epoch {target_epoch}...")
        print("="*60)
        
        # Train for the remaining epochs
        trainer = pl.Trainer(
            default_root_dir=f"temp_epoch_{target_epoch}",
            accelerator="auto",
            max_epochs=target_epoch,
            enable_progress_bar=True,
            logger=False
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"epoch_{target_epoch}.ckpt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Generate samples
        generate_and_save_samples(model, f"epoch_{target_epoch}", save_dir)
    
    
    # ===== 3. Optional: Also load and sample from the previously trained model =====
    print("\n" + "="*60)
    print("Loading previously trained model (Epoch 76) for comparison...")
    print("="*60)
    
    checkpoint_path_76 = "VAE_logs/lightning_logs/version_2/checkpoints/epoch=76-step=32494.ckpt"
    
    if os.path.exists(checkpoint_path_76):
        model_epoch76 = VAE.load_from_checkpoint(checkpoint_path_76)
        generate_and_save_samples(model_epoch76, "epoch_76_pretrained", save_dir)
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path_76}")
    
    
    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"All sample images saved in: {save_dir}/")
    print("\nGenerated files:")
    print("  - epoch_0_untrained_samples.png   (before training)")
    print("  - epoch_1_samples.png             (after 1 epoch)")
    print("  - epoch_5_samples.png             (after 5 epochs)")
    print("  - epoch_10_samples.png            (after 10 epochs)")
    print("  - epoch_20_samples.png            (after 20 epochs)")
    print("  - epoch_80_samples.png            (after 80 epochs)")
    print("  - epoch_76_pretrained_samples.png (previous training for comparison)")
    print("\nNow you can analyze the quality improvement across these stages!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
