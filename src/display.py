import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from src.utils import jax_to_numpy, world_to_voxel

def plot_slice(ax, nifti, slice=65, cmap='gray', axis=0):
    volume = jnp.array(nifti.get_fdata())
    affine = nifti.affine

    # Accquire the slice in world coordinates
    if axis == 0:
        world_coord = np.array([slice, 0, 0])
    elif axis == 1:
        world_coord = np.array([0, slice, 0])
    elif axis == 2:
        world_coord = np.array([0, 0, slice])

    # Convert world coordinates to voxel coordinates
    voxel_coord = world_to_voxel(world_coord, affine)

    # Get the slice in voxel coordinates
    if axis == 0:
        slice = int(voxel_coord[0])
    elif axis == 1:
        slice = int(voxel_coord[1])
    elif axis == 2:
        slice = int(voxel_coord[2])
    
    # Get the slice from the volume
    if axis == 0:
        volume = volume[slice, :, :]
    elif axis == 1:
        volume = volume[:, slice, :]
    elif axis == 2:
        volume = volume[:, :, slice]

    ax.imshow(jnp.abs(volume), cmap=cmap)

def display_comparison(niftis, slice=24, axis=0):
    fig = plt.figure(figsize=(15, 10))

    for i, nifti in enumerate(niftis):
        ax = fig.add_subplot(1, len(niftis), i + 1)
        plot_slice(ax, nifti, slice=slice, axis=axis)
        ax.set_title(f"Image {i+1}")

    # plot_slice(axes[0], nifti_1, slice=slice, axis=axis)
    # axes[0].set_title('Image 1')
    
    # plot_slice(axes[1], nifti_2, slice=slice, axis=axis)
    # axes[1].set_title('Image 2')

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
        data_slice = jnp.abs(volume[slice_idx, :, :])  # Take absolute to handle complex values
    elif axis == 1:
        data_slice = jnp.abs(volume[:, slice_idx, :])
    elif axis == 2:
        data_slice = jnp.abs(volume[:, :, slice_idx])

    data_slice = jax_to_numpy(data_slice)  # Convert JAX array to NumPy array

    X, Y = np.meshgrid(np.arange(data_slice.shape[1]), np.arange(data_slice.shape[0]))  # Create grid
    ax.plot_surface(X, Y, data_slice, cmap=cmap, edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity")
    ax.view_init(elev=30, azim=135)  # Adjust viewing angle
    if limit > 0:
        ax.set_zlim(0, limit)