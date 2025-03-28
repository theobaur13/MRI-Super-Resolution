import os
import nibabel as nib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import SimpleITK as sitk
from tqdm import tqdm

def get_brats_paths(data_dir, seq, dataset):
    train_dir = os.path.join(data_dir, dataset, "train")
    validate_dir = os.path.join(data_dir, dataset, "validate")

    train_paths = [os.path.join(train_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(train_dir)]
    validate_paths = [os.path.join(validate_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(validate_dir)]

    return train_paths, validate_paths

def get_picai_paths(data_dir, fold, seq, limit=1):
    dir = os.path.join(data_dir, "images", f"fold{fold}")

    paths = []
    for patient in tqdm(os.listdir(dir)):
        patient_dir = os.path.join(dir, patient)
        for file in os.listdir(patient_dir):
            if file.endswith(f"{seq}.mha"):
                paths.append(os.path.join(patient_dir, file))
                if len(paths) == limit:
                    return paths
    return paths

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def read_metaimage(file_path):
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

def plot_matrix(ax, matrix, slice=10, cmap='gray', axis="z"):
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

def display(image, kspace, simulated_kspace, simulated_image, axis=0, highlight=False):
    fig_images, axes_images = plt.subplots(1, 2, figsize=(9, 4))
    
    plot_matrix(axes_images[0], image, axis=axis)
    axes_images[0].set_title('Original Image')
    
    plot_matrix(axes_images[1], simulated_image, axis=axis)
    axes_images[1].set_title('Simulated Image')

    plt.tight_layout()
    plt.show(block=False)  

    fig_kspace, axes_kspace = plt.subplots(1, 2, figsize=(9, 4))

    plot_matrix(axes_kspace[0], kspace, axis=axis)
    axes_kspace[0].set_title('Original k-space')
    
    if highlight:
        simulated_kspace = jnp.where(simulated_kspace == 0, jnp.nan, simulated_kspace)
    plot_matrix(axes_kspace[1], simulated_kspace, axis=axis)
    axes_kspace[1].set_title('Simulated k-space')
    
    plt.tight_layout()
    plt.show()