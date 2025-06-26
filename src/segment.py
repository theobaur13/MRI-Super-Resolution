import os
import shutil
import subprocess
from tqdm import tqdm
from src.utils.paths import get_brats_paths

def segment(args):
    paths, _ = get_brats_paths(args.dataset_dir)

    flywheel_dir = os.path.join(os.getcwd(), "flywheel", "v0")
    flywheel_input_dir = os.path.join(flywheel_dir, "input", "nifti")
    flywheel_output_dir = os.path.join(flywheel_dir, "output")
    config_path = os.path.join(flywheel_dir, "config.json")

    for i in tqdm(range(args.limit)):
        # Skip condition
        scan_path = paths[i]
        scan_dir = os.path.dirname(scan_path)
        scan_name = os.path.basename(scan_path).split(".")[0]
        target = scan_name + "_fast_mixeltype.nii.gz"
        seq = scan_name.split("-")[-1]
        if seq in ["t1c", "t1n"]:
            n = "1"
        elif seq in ["t2f", "t2w"]:
            n = "2"
            
        print(f"Segmenting {scan_path} with sequence {seq}...")
        if os.path.exists(os.path.join(scan_dir, target)):
            print(f"Skipping {scan_path} as it has already been segmented.")
            continue

        shutil.copy(scan_path, flywheel_input_dir)

        subprocess.run([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{config_path}:/flywheel/v0/config.json",
            "-v", f"{flywheel_input_dir}:/flywheel/v0/input/nifti",
            "-v", f"{flywheel_output_dir}:/flywheel/v0/output",
            "scitran/fsl-fast",
            "-t", n
        ])

        for file in os.listdir(flywheel_output_dir):
            if file.endswith(".nii.gz"):
                shutil.copy(os.path.join(flywheel_output_dir, file), os.path.join(scan_dir, file))

        # Clear input and output directories
        os.remove(os.path.join(flywheel_input_dir, os.path.basename(scan_path)))
        for file in os.listdir(flywheel_output_dir):
            if file != '.gitkeep':
                os.remove(os.path.join(flywheel_output_dir, file))