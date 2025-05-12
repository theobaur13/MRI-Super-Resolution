import torch
import jax.numpy as jnp
import numpy as np

def jax_to_numpy(x):
    """Convert a JAX array to a NumPy array."""
    if isinstance(x, jnp.ndarray):
        return np.array(x)
    else:
        raise TypeError("Input must be a JAX array.")

def numpy_to_jax(x):
    """Convert a NumPy array to a JAX array."""
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    else:
        raise TypeError("Input must be a NumPy array.")

def convert_to_tensor(image_list, slice_axis):
    real_slices = []
    imag_slices = []

    for img in image_list:
        img_np = jnp.array(img)
        for i in range(img_np.shape[slice_axis]):
            slice_2d = img_np[i]
            real_slices.append(slice_2d.real)
            imag_slices.append(slice_2d.imag)
    
    real_slices = jnp.array(real_slices)
    imag_slices = jnp.array(imag_slices)

    real_tensor = torch.tensor(real_slices, dtype=torch.float32).unsqueeze(1)
    imag_tensor = torch.tensor(imag_slices, dtype=torch.float32).unsqueeze(1)
    return torch.cat((real_tensor, imag_tensor), dim=1)

def robust_max(image, axis, slice_idx=None):
    if axis == 0:
        data = image[slice_idx, :, :]
    elif axis == 1:
        data = image[:, slice_idx, :]
    elif axis == 2:
        data = image[:, :, slice_idx]
    
    # Flatten the data to a 1D array for statistical operations
    data_flat = jnp.abs(data.flatten())
    
    # Calculate the 25th and 75th percentiles (Q1 and Q3)
    q1 = jnp.percentile(data_flat, 40)
    q3 = jnp.percentile(data_flat, 60)
    
    # Calculate the IQR (Interquartile Range)
    iqr = q3 - q1
    
    # Define bounds for removing outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out values that are outside the IQR range
    filtered_data = data_flat[(data_flat >= lower_bound) & (data_flat <= upper_bound)]
    
    # Calculate the maximum of the filtered data
    return filtered_data.max()

def world_to_voxel_coords(world_coord, affine):
    """Convert real-world (mm) coordinates to voxel indices."""
    return np.round(np.linalg.inv(affine) @ np.append(world_coord, 1))[:3].astype(int)

def world_to_voxel_slice(slice_idx, axis, affine):
        # Accquire the slice in world coordinates
    if axis == 0:
        world_coord = np.array([slice_idx, 0, 0])
    elif axis == 1:
        world_coord = np.array([0, slice_idx, 0])
    elif axis == 2:
        world_coord = np.array([0, 0, slice_idx])

    # Convert world coordinates to voxel coordinates
    voxel_coord = world_to_voxel_coords(world_coord, affine)

    # Get the slice in voxel coordinates
    if axis == 0:
        slice_num = int(voxel_coord[0])
    elif axis == 1:
        slice_num = int(voxel_coord[1])
    elif axis == 2:
        slice_num = int(voxel_coord[2])
    return slice_num

def slice_nifti(nifti, slice_idx, axis):
    volume = jnp.array(nifti.get_fdata())
    affine = nifti.affine
    
    # Convert the slice index to voxel coordinates
    slice_num = world_to_voxel_slice(slice_idx, axis, affine)

    # Get the slice from the volume
    if axis == 0:
        slice = volume[slice_num, :, :]
    elif axis == 1:
        slice = volume[:, slice_num, :]
    elif axis == 2:
        slice = volume[:, :, slice_num]

    return slice