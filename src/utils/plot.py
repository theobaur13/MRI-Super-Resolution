import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import pandas as pd
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


def plot_training_log(csv_file, output_file):
    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    if not {"epoch", "batch"}.issubset(df.columns):
        raise ValueError("CSV file must contain 'epoch' and 'batch' columns.")

    # Get all metric columns (everything except 'epoch' and 'batch')
    metric_cols = [col for col in df.columns if col not in {"epoch", "batch"}]
    if len(metric_cols) == 0:
        raise ValueError("CSV file must contain at least one metric column.")

    # Sort by epoch and batch
    df = df.sort_values(by=["epoch", "batch"]).reset_index(drop=True)

    # Create step number for x-axis
    df["step"] = range(len(df))

    # Plotting
    plt.figure(figsize=(12, 6))
    for col in metric_cols:
        plt.plot(df["step"], df[col], label=col)

    plt.xlabel("Step (epoch + batch)")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Over Time")

    # Add vertical red lines for epoch boundaries
    epoch_starts = df.groupby("epoch")["step"].min().tolist()
    for step in epoch_starts[1:]:  # skip first epoch at step 0
        plt.axvline(x=step, color="red", linestyle="--", linewidth=1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_lr_sr_hr(lr, sr, hr):
    # Plotting the slices
    vmin = min(lr.min(), sr.min(), hr.min())
    vmax = max(lr.max(), sr.max(), hr.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Low Resolution Slice')
    axes[0].axis('off')

    axes[1].imshow(sr, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Super Resolution Slice')
    axes[1].axis('off')

    axes[2].imshow(hr, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('High Resolution Slice')
    axes[2].axis('off')
    
    return fig