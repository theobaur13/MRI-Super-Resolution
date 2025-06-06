import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from src.utils.conversions import jax_to_numpy
from src.utils.slicing import slice_nifti

# Extract a 2D slice from a 3D volume or NIfTI object.
def extract_slice(data, slice=65, axis=0):
    if hasattr(data, "get_fdata"):
        # If data is a NIfTI object, use the slice_nifti function
        return slice_nifti(data, slice_idx=slice, axis=axis)
    
    # If data is a raw volume, extract the slice directly
    volume = jnp.array(data)
    if axis == 0:
        return volume[slice, :, :]
    elif axis == 1:
        return volume[:, slice, :]
    elif axis == 2:
        return volume[:, :, slice]

# Generic slice plotter for volume or NIfTI.
def plot_slice(ax, data, slice=65, axis=0, cmap='gray'):
    slice_data = extract_slice(data, slice=slice, axis=axis)
    ax.imshow(jnp.abs(slice_data), cmap=cmap)

# Display a comparison of NIfTI or raw volumes."""
def display_img(data_list, slice=24, axis=0, titles=None):
    fig = plt.figure(figsize=(15, 10))

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(1, len(data_list), i + 1)
        plot_slice(ax, data, slice=slice, axis=axis)
        ax.set_title(titles[i] if titles else f"Image {i+1}")

    plt.tight_layout()
    plt.show(block=False)

# Display a 3D surface plot of the volume or NIfTI object.
def display_3d(data, slice=65, axis=0, limit=1, titles=None):
    fig = plt.figure(figsize=(15, 10))

    for i, volume in enumerate(data):
        ax = fig.add_subplot(1, len(data), i + 1, projection='3d')
        plot_surface(ax, volume, slice=slice, axis=axis, limit=limit)
        ax.set_title(titles[i] if titles else f"Image {i+1}")

    plt.tight_layout()
    plt.show(block=False)

# Display a 3D surface plot of the volume or NIfTI object.
def plot_surface(ax, data, slice=65, axis=0, cmap="plasma", limit=0):
    slice_data = extract_slice(data, slice=slice, axis=axis)
    numpy_slice = np.abs(jax_to_numpy(slice_data))  # Convert JAX array to NumPy array

    X, Y = np.meshgrid(np.arange(numpy_slice.shape[1]), np.arange(numpy_slice.shape[0]))  # Create grid
    ax.plot_surface(X, Y, numpy_slice, cmap=cmap, edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity")
    ax.set_zscale("log")  # Set Z-axis to logarithmic scale
    ax.view_init(elev=30, azim=135)  # Adjust viewing angle
    if limit > 0:
        min = jnp.min(numpy_slice[numpy_slice > 0]) if jnp.any(numpy_slice > 0) else 1e-3
        ax.set_zlim(min, limit)

def plot_training_log(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)
    df.ffill(inplace=True)

    # Determine where to save plots
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # --- Plot 1: Generator vs Discriminator Loss ---
    plt.figure(figsize=(20, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df['batch'], df['gen_loss'], label='Generator Loss', color='blue')
    ax2.plot(df['batch'], df['disc_loss'], label='Discriminator Loss', color='red')
    ax1.set_ylabel('Generator Loss', color='blue')
    ax2.set_ylabel('Discriminator Loss', color='red')
    ax1.set_xlabel('Batch')
    plt.title("Generator vs Discriminator Loss")
    for epoch in sorted(df['epoch'].dropna().unique()):
        epoch_rows = df[df['epoch'] == epoch]
        if not epoch_rows.empty:
            batch_start = epoch_rows['batch'].iloc[0]
            ax1.axvline(x=batch_start, color='gray', linestyle=':', linewidth=0.8)
            ax1.text(batch_start, ax1.get_ylim()[1] * 0.95, f"Epoch {int(epoch)}", rotation=90,
                    fontsize=8, color='gray')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # --- Plot 2: Train and Validation PSNR ---
    plt.figure(figsize=(12, 5))
    plt.plot(df['batch'], df['train_psnr'], label='Train PSNR', linestyle='--')
    plt.plot(df['batch'], df['val_psnr'], label='Validation PSNR', linestyle='-')
    for epoch in sorted(df['epoch'].dropna().unique()):
        epoch_rows = df[df['epoch'] == epoch]
        if not epoch_rows.empty:
            batch_start = epoch_rows['batch'].iloc[0]
            plt.axvline(x=batch_start, color='gray', linestyle=':', linewidth=0.8)
            plt.text(batch_start, plt.ylim()[1] * 0.95, f"Epoch {int(epoch)}", rotation=90, fontsize=8, color='gray')
    plt.title("Train and Validation PSNR")
    plt.xlabel("Batch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_plot.png"))
    plt.close()

    # --- Plot 3: Train and Validation SSIM ---
    plt.figure(figsize=(12, 5))
    plt.plot(df['batch'], df['train_ssim'], label='Train SSIM', linestyle='--')
    plt.plot(df['batch'], df['val_ssim'], label='Validation SSIM', linestyle='-')
    for epoch in sorted(df['epoch'].dropna().unique()):
        epoch_rows = df[df['epoch'] == epoch]
        if not epoch_rows.empty:
            batch_start = epoch_rows['batch'].iloc[0]
            plt.axvline(x=batch_start, color='gray', linestyle=':', linewidth=0.8)
            plt.text(batch_start, plt.ylim()[1] * 0.95, f"Epoch {int(epoch)}", rotation=90, fontsize=8, color='gray')
    plt.title("Train and Validation SSIM")
    plt.xlabel("Batch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssim_plot.png"))
    plt.close()