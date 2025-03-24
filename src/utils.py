import nibabel as nib
import matplotlib.pyplot as plt

def read_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def plot_nifti(matrix):
    plt.imshow(matrix, cmap='gray')
    plt.show()