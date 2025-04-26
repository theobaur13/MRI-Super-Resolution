import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_slice(ax, volume, slice=65, cmap='gray', axis=0):
    # set NaN values to red on colormap
    cmap = plt.cm.gray
    cmap.set_bad(color='red')
    
    volume = volume.real
    if axis == 0:
        volume = volume[slice, :, :]
    elif axis == 1:
        volume = volume[:, slice, :]
    elif axis == 2:
        volume = volume[:, :, slice]
        
    ax.imshow(jnp.abs(volume), cmap=cmap)
    # ax.axis('off')

def display_comparison(volume_1, volume_2, slice=24, axis=0, kspace=True):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    plot_slice(axes[0], volume_1, slice=slice, axis=axis)
    axes[0].set_title('Image 1')

    if kspace:
        jnp.where(volume_2 == 0, jnp.nan, volume_2)
    plot_slice(axes[1], volume_2, slice=slice, axis=axis)
    axes[1].set_title('Image 2')

    plt.tight_layout()
    plt.show(block=False)

def plot_3d_surfaces(volumes, slice_idx, axis=0, cmap="viridis", limit=0):
    fig = plt.figure(figsize=(15, 10))

    for i, volume in enumerate(volumes):
        ax = fig.add_subplot(1, len(volumes), i + 1, projection='3d')
        plot_surface(ax, volume, slice_idx, axis=axis, cmap=cmap, limit=limit)
        ax.set_title(f"K-space {i+1}")

    plt.tight_layout()
    plt.show(block=False)

def plot_surface(ax, volume, slice_idx, axis=0, cmap="viridis", limit=0):
    """Plots a 3D surface of a 2D slice from a 3D matrix."""
    if axis == 0:
        data_slice = np.abs(volume[slice_idx, :, :])  # Take absolute to handle complex values
    elif axis == 1:
        data_slice = np.abs(volume[:, slice_idx, :])
    elif axis == 2:
        data_slice = np.abs(volume[:, :, slice_idx])

    X, Y = np.meshgrid(np.arange(data_slice.shape[1]), np.arange(data_slice.shape[0]))  # Create grid
    ax.plot_surface(X, Y, data_slice, cmap=cmap, edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity")
    ax.view_init(elev=30, azim=135)  # Adjust viewing angle
    if limit > 0:
        ax.set_zlim(0, limit)