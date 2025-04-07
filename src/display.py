import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_matrix(ax, matrix, slice=65, cmap='gray', axis="z"):
    # set NaN values to red on colormap
    cmap = plt.cm.gray
    cmap.set_bad(color='red')
    
    matrix = matrix.real
    if axis == 0:
        matrix = matrix[slice, :, :]
    elif axis == 1:
        matrix = matrix[:, slice, :]
    elif axis == 2:
        matrix = matrix[:, :, slice]
        
    ax.imshow(matrix, cmap=cmap)
    # ax.axis('off')

def display_comparison(image_1, image_2, slice=24, axis=0, kspace=True):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    plot_matrix(axes[0], image_1, slice=slice, axis=axis)
    axes[0].set_title('Image 1')

    if kspace:
        jnp.where(image_2 == 0, jnp.nan, image_2)
    plot_matrix(axes[1], image_2, slice=slice, axis=axis)
    axes[1].set_title('Image 2')

    plt.tight_layout()
    plt.show(block=False)

def plot_3d_kspace(kspaces, slice_idx, axis=0, cmap="viridis", limit=True):
    fig = plt.figure(figsize=(15, 10))

    for i, kspace in enumerate(kspaces):
        ax = fig.add_subplot(1, len(kspaces), i + 1, projection='3d')
        plot_surface(ax, kspace, slice_idx, axis=axis, cmap=cmap, limit=limit)
        ax.set_title(f"K-space {i+1}")

    plt.tight_layout()
    plt.show(block=False)

def plot_surface(ax, matrix, slice_idx, axis=0, cmap="viridis", limit=True):
    """Plots a 3D surface of a 2D slice from a 3D matrix."""
    if axis == 0:
        data_slice = np.abs(matrix[slice_idx, :, :])  # Take absolute to handle complex values
    elif axis == 1:
        data_slice = np.abs(matrix[:, slice_idx, :])
    elif axis == 2:
        data_slice = np.abs(matrix[:, :, slice_idx])

    X, Y = np.meshgrid(np.arange(data_slice.shape[1]), np.arange(data_slice.shape[0]))  # Create grid
    ax.plot_surface(X, Y, data_slice, cmap=cmap, edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity")
    ax.view_init(elev=30, azim=135)  # Adjust viewing angle
    if limit:
        ax.set_zlim(0, 50000)