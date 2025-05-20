import os
import argparse
from src.cli import (
    convert_adni,
    simulate,
    analyse,
    batch_convert,
    view
)

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Subparser for converting ADNI data
    # > py main.py convert-adni --ADNI_dir "data/ADNI" --ADNI_nifti_dir "data/ADNI_NIfTIs"
    convert_parser = subparsers.add_parser("convert-adni", help="Convert ADNI data to NIfTI format")
    convert_parser.add_argument("--ADNI_dir", type=str, help="Path to ADNI directory")
    convert_parser.add_argument("--ADNI_nifti_dir", type=str, help="Path to ADNI NIfTI directory")

    # Subparser for simulating data
    # > py main.py simulate --path "data/ADNI_NIfTIs/3T/ADNI_002_S_0413_MR_Double_TSE_br_raw_20061115141733_1_S22682_I30117.nii.gz" --axis 2 --slice 24
    simulate_parser = subparsers.add_parser("simulate", help="Simulate data")
    simulate_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to simulate")
    simulate_parser.add_argument("--axis", type=int, default=0, help="Axis for simulation")
    simulate_parser.add_argument("--slice", type=int, default=24, help="Slice index for simulation")

    # Subparser for analysing noise
    # > py main.py analyse-snr-avg --dataset "ADNI" --axis 2
    # > py main.py analyse-snr-avg --dataset "BraTS" --axis 2
    analyse_noise_parser = subparsers.add_parser("analyse-snr-avg", help="Analyse noise in data")
    analyse_noise_parser.add_argument("--dataset", type=str, required=True, help="Dataset for analysis (e.g., 'ADNI', 'BraTS')")
    analyse_noise_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")

    # Subparser for analysing SNR map
    # > py main.py analyse-snr-map --dataset "ADNI" --axis 2 --slice 24
    # > py main.py analyse-snr-map --dataset "BraTS" --axis 2 --slice 65
    analyse_snr_map_parser = subparsers.add_parser("analyse-snr-map", help="Analyse SNR map in data")
    analyse_snr_map_parser.add_argument("--dataset", type=str, required=True, help="Dataset for analysis (e.g., 'ADNI', 'BraTS')")
    analyse_snr_map_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_snr_map_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")

    # Subparser for analysing brightness
    # > py main.py analyse-brightness --dataset "ADNI" --slice 24 --axis 2
    # > py main.py analyse-brightness --dataset "BraTS" --slice 65 --axis 2
    analyse_brightness_parser = subparsers.add_parser("analyse-brightness", help="Analyse brightness in data")
    analyse_brightness_parser.add_argument("--dataset", type=str, required=True, help="Dataset for analysis (e.g., 'ADNI', 'BraTS')")
    analyse_brightness_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_brightness_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")

    # Subparser for batch conversion
    # > py main.py batch-convert --seq "t1c" --brats_dataset "BraSyn" --output_dir "data/BraTS_output"
    batch_convert_parser = subparsers.add_parser("batch-convert", help="Batch convert data")
    batch_convert_parser.add_argument("--seq", type=str, required=True, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    batch_convert_parser.add_argument("--brats_dataset", type=str, required=True, help="Dataset type (e.g., 'BraSyn', 'GLI')")
    batch_convert_parser.add_argument("--output_dir", type=str, help="Output directory for converted data")

    # Subparser for viewing data
    # > py main.py view --path "data/BraTS_output/BraTS-GLI-00000-000-t2f.nii.gz" --slice 65 --axis 2
    # > py main.py view --path "data/data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000/BraSyn/train/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz" --slice 65 --axis 2
    # > py main.py view --path "data/ADNI_NifTIs/1.5T/ADNI_002_S_0413_MR_Axial_PD_T2_FSE__br_raw_20061115094759_1_S22556_I29704.nii.gz" --slice 24 --axis 2
    view_parser = subparsers.add_parser("view", help="View NIfTI data")
    view_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to view")
    view_parser.add_argument("--slice", type=int, default=24, help="Slice index for viewing")
    view_parser.add_argument("--axis", type=int, default=0, help="Axis for viewing")
    
    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    if action == "convert-adni":
        convert_adni(args, base_dir)

    # Apply degradation to slice in a volume
    elif action == "simulate":
        simulate(args, base_dir)

    # Perform analysis between two types of scans
    elif action == "analyse-snr-avg" or action == "analyse-brightness" or action == "analyse-snr-map":
        analyse(args, base_dir)

    # Apply degradation to BraTS scans
    elif action == "batch-convert":
        batch_convert(args, base_dir)

    # View a slice of a NIfTI file
    elif action == "view":
        view(args, base_dir)