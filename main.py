import os
from src.utils import read_nifti, plot_nifti

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    seq = "t2w"
    data_path = os.path.join(base_path, "data", "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000", "BraSyn", "train", "BraTS-GLI-00000-000", f"BraTS-GLI-00000-000-{seq}.nii.gz")

    nifti = read_nifti(data_path)
    nifti_slice = nifti[:, :, 50]
    plot_nifti(nifti_slice)