import numpy as np
import jax.numpy as jnp

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