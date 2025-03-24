import os
import nibabel as nib
import matplotlib.pyplot as plt

def get_paths(data_dir, seq, dataset):
    train_dir = os.path.join(data_dir, dataset, "train")
    validate_dir = os.path.join(data_dir, dataset, "validate")

    train_paths = [os.path.join(train_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(train_dir)]
    validate_paths = [os.path.join(validate_dir, patient, f"{patient}-{seq}.nii.gz") for patient in os.listdir(validate_dir)]

    return train_paths, validate_paths

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def plot_matrix(matrix, slice=75, cmap='gray', axis="z"):
    matrix = matrix.real
    if axis == 0:
        matrix = matrix[slice, :, :]
    elif axis == 1:
        matrix = matrix[:, slice, :]
    elif axis == 2:
        matrix = matrix[:, :, slice]
        
    plt.imshow(matrix, cmap=cmap)
    plt.show()