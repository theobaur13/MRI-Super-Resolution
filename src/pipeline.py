import jax.numpy as jnp
import nibabel as nib
from src.display import display_img, display_3d
from src.slicing import world_to_voxel_slice
from src.conversions import (
    convert_to_kspace,
    convert_to_image,
    jax_to_numpy,
)
from src.transformations import (
    cylindrical_crop,
    gaussian_amplification,
    random_noise
)

def simluation_pipeline(nifti, axis, visualize=False, slice=None):
    # Degrade 3T scans to resemble 1.5T scans
    original_volume = jnp.array(nifti.get_fdata())
    original_kspace = convert_to_kspace(original_volume)

    # k-space manipulation
    cylindrical_cropped_kspace = cylindrical_crop(original_kspace, axis=axis, factor=0.7)
    gaussian_amped_kspace = gaussian_amplification(cylindrical_cropped_kspace, axis=0, sigma=0.5, mu=0.5, A=2)
    
    # Image manipulation
    simulated_volume = convert_to_image(gaussian_amped_kspace)
    # gibbs_reduced_volume = numpy_to_jax(gibbs_removal(jax_to_numpy(simulated_volume), slice_axis=axis))
    gaussian_amped_image = gaussian_amplification(simulated_volume, axis=0, sigma=0.4, mu=0.5, A=1, invert=True)
    noisy_image = random_noise(gaussian_amped_image, intensity=0.01, frequency=0.3)
    
    if visualize:
        if slice is None:
            raise ValueError("Slice index must be provided for visualization.")
        
        # Calculate the correct slice index for the given axis
        real_world_slice = slice
        voxel_slice = world_to_voxel_slice(real_world_slice, axis, nifti.affine)

        # Display the original and simulated images
        display_img([
            original_volume, simulated_volume, gaussian_amped_image, noisy_image], 
            slice=voxel_slice, axis=axis,
            titles=["Original", "k-Space Manipulation", "Central Brightening", "Noise"])
        
        # Display the k-space
        display_3d(
            [original_kspace, cylindrical_cropped_kspace, gaussian_amped_kspace], 
            slice=voxel_slice, axis=axis, limit=1,
            titles=["Original k-Space", "Cylindrical Crop", "Gaussian Amplification"])

    # Convert to NIfTI
    simulated_nifti = nib.Nifti1Image(jax_to_numpy(noisy_image), affine=nifti.affine)
    return simulated_nifti, gaussian_amped_kspace