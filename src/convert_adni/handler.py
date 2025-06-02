import os
from tqdm import tqdm
import dicom2nifti
from src.utils.paths import get_adni_paths

def convert_adni(args):
    ADNI_dir = args.ADNI_dir
    output_dir = args.ADNI_nifti_dir

    t1_5_paths, t3_paths = get_adni_paths(ADNI_dir)

    # Create 1.5T and 3T subdirectories
    os.makedirs(os.path.join(output_dir, "1.5T"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "3T"), exist_ok=True)

    def process(paths, target_dir):
        for dir in tqdm(paths):
            # Edit timestamps to reflect the directory name
            timestamp = dir.split("\\")[-2]
            timestamp = timestamp.replace(".0", "")
            timestamp = timestamp.replace("_", "")
            timestamp = timestamp.replace("-", "")
            contents = os.listdir(dir)
            
            old_name = contents[0]
            parts = old_name.split("_")
            parts[-4] = timestamp
            new_name = "_".join(parts)
            new_name = new_name.replace(".dcm", ".nii.gz")

            # Convert the volume to a NIfTI image
            dicom2nifti.dicom_series_to_nifti(dir, os.path.join(target_dir, new_name), reorient_nifti=True)

    process(t1_5_paths, os.path.join(output_dir, "1.5T"))
    process(t3_paths, os.path.join(output_dir, "3T"))
