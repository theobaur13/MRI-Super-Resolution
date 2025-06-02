import jax
import jax.numpy as jnp
import nibabel as nib
from src.display.plot import display_img, display_3d
from src.utils.slicing import world_to_voxel_slice
from src.utils.conversions import (
    convert_to_kspace,
    convert_to_image,
    jax_to_numpy,
    numpy_to_jax
)
from src.simulation.transformations import (
    variable_density_undersampling,
    radial_undersampling,
    spiral_undersampling,
    cartesian_undersampling,
    cylindrical_crop,
    gaussian_amplification,
    rician_noise,
    matter_noise,
    partial_fourier,
    matter_contrast
)

def core(image: jax.Array, axis: int) -> tuple[dict, dict]:
    """
    Core function to apply transformations to the image.
    
    Args:
        image (jax.Array): Input image in JAX array format.
        axis (int): Axis along which to apply transformations.
    
    Returns:
        dict: Dictionary containing intermediate results of the transformations.
    """
    kspaces = {}
    images = {}

    # image = matter_contrast(image, path)
    # images["matter_contrast"] = image

    ### === k-Space Domain === ###
    kspace = convert_to_kspace(image)
    kspaces["original"] = kspace

    kspace = cylindrical_crop(kspace, axis=axis, factor=0.58)
    kspaces["cylindrical_crop"] = kspace

    kspace = radial_undersampling(kspace, axis=axis)
    kspaces["radial_undersampling"] = kspace

    # kspace = spiral_undersampling(kspace, axis=axis)
    # kspaces["spiral_undersampling"] = kspace

    # kspace = cartesian_undersampling(kspace, axis=axis)
    # kspaces["cartesian_undersampling"] = kspace

    # kspace = variable_density_undersampling(kspace, density=0.5, steepness=15)
    # kspaces["variable_density_undersampling"] = kspace

    kspace = partial_fourier(kspace, axis=axis, fraction=0.625)
    kspaces["partial_fourier"] = kspace

    ### === Image Domain === ###
    image = convert_to_image(kspace)
    images["k_space_manipulation"] = image

    image = gaussian_amplification(image, axis=0, spread=0.5, centre=0.5, amplitude=0.7, invert=True)
    images["central_brightening"] = image

    image = rician_noise(image, base_noise=0.005)
    images["rician_noise"] = image

    images["final"] = image

    return images, kspaces

core = jax.jit(core, static_argnames=["axis"])

def simluation_pipeline(nifti, axis, path, visualize=False, slice=None):
    ### === Image Domain === ###
    image = jnp.array(nifti.get_fdata())
    images = {"original": image}

    images, kspaces = core(image, axis)

    # Visualise each stage of the simulation pipeline if requested
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
        
        # Display the original and simulated k-spaces
        display_3d(
            list(kspaces.values()),
            slice=voxel_slice, axis=axis, limit=1,
            titles=list(kspaces.keys()))

    # Convert the final image to NIfTI format
    simulated_nifti = nib.Nifti1Image(jax_to_numpy(images["final"]), affine=nifti.affine)
    return simulated_nifti

# Simulate a batch of images
def simulate_batch(images, axis: int):
    @jax.jit
    def simulate_one(image):
        return core(image, axis)
    return jax.vmap(simulate_one)(images)