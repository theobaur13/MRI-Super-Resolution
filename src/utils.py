import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def get_paths(data_dir, seq, dataset):
    train_dir = os.path.join(data_dir, dataset, "train")
    validate_dir = os.path.join(data_dir, dataset, "validate")

    train_paths = [os.path.join(train_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(train_dir)]
    validate_paths = [os.path.join(validate_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(validate_dir)]

    return train_paths, validate_paths

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def plot_matrix(ax, matrix, slice=75, cmap='gray', axis="z"):
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
    ax.axis('off')

def display(image, kspace, simulated_kspace, simulated_image, axis=0, highlight=False):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    
    plot_matrix(axes[0, 0], image, axis=axis)
    axes[0, 0].set_title('Original Image')
    
    plot_matrix(axes[0, 1], kspace, axis=axis)
    axes[0, 1].set_title('Original k-space')
    
    if highlight:
        simulated_kspace[simulated_kspace == 0] = np.nan
    plot_matrix(axes[1, 1], simulated_kspace, axis=axis)
    axes[1, 1].set_title('Simulated k-space')
    
    plot_matrix(axes[1, 0], simulated_image, axis=axis)
    axes[1, 0].set_title('Simulated Image')
    
    plt.tight_layout()
    plt.show()