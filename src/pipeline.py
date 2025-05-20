import jax.numpy as jnp
import nibabel as nib
from src.display import display_img, display_3d
from src.slicing import world_to_voxel_slice
from src.conversions import (
    convert_to_kspace,
    convert_to_image,
    jax_to_numpy,
    numpy_to_jax
)
from src.transformations import (
    variable_density_undersampling,
    cylindrical_crop,
    gaussian_amplification,
    rician_noise,
    rician_edge_noise,
    partial_fourier
)

from src.gibbs_removal import gibbs_removal

def simluation_pipeline(nifti, axis, visualize=False, slice=None):
    # Degrade 3T scans to resemble 1.5T scans
    image = jnp.array(nifti.get_fdata())
    kspace = convert_to_kspace(image)

    # Create dictionaries to store volumes and k-space data
    images = {"original": image}
    kspaces = {"original": kspace}

    # k-space manipulation
    kspace = variable_density_undersampling(kspace)
    kspaces["variable_density_undersampling"] = kspace

    kspace = cylindrical_crop(kspace, axis=axis, factor=0.7)
    kspaces["cylindrical_crop"] = kspace

    # kspace = gaussian_amplification(kspace, axis=0, spread=0.5, centre=0.5, amplitude=2)
    # kspaces["gaussian_amplification"] = kspace

    kspace = partial_fourier(kspace, axis=axis, fraction=0.625)
    kspaces["partial_fourier"] = kspace
    
    # Image manipulation
    image = convert_to_image(kspace)
    images["k_space_manipulation"] = image

    # image = numpy_to_jax(gibbs_removal(jax_to_numpy(image), slice_axis=axis))
    # images["gibbs_reduction"] = image

    image = gaussian_amplification(image, axis=0, spread=0.5, centre=0.5, amplitude=0.8, invert=True)
    images["central_brightening"] = image

    # image = rician_noise(image, base_noise=0.01)
    # images["rician_noise"] = image
    
    image = rician_edge_noise(image, axis=axis, base_noise=0.1, edge_strength=0.1)
    images["rician_edge_noise"] = image

    if visualize:
        if slice is None:
            raise ValueError("Slice index must be provided for visualization.")
        
        # Calculate the correct slice index for the given axis
        real_world_slice = slice
        voxel_slice = world_to_voxel_slice(real_world_slice, axis, nifti.affine)

        # Display the original and simulated images
        display_img(
            list(images.values()), 
            slice=voxel_slice, axis=axis,
            titles=list(images.keys()))
        
        # Display the k-space
        display_3d(
            list(kspaces.values()),
            slice=voxel_slice, axis=axis, limit=1,
            titles=list(kspaces.keys()))

    # Convert to NIfTI
    simulated_nifti = nib.Nifti1Image(jax_to_numpy(image), affine=nifti.affine)
    return simulated_nifti, kspace