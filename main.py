import os
import argparse
from src.cli import (
    convert_adni,
    simulate,
    analyse,
    batch_simulate,
    view,
    segment
)

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Subparser for converting ADNI data
    # > py main.py convert-adni --ADNI_dir "D:\ADNI" --ADNI_nifti_dir "D:\ADNI_NIfTIs"
    convert_parser = subparsers.add_parser("convert-adni", help="Convert ADNI data to NIfTI format")
    convert_parser.add_argument("--ADNI_dir", type=str, help="Path to ADNI directory")
    convert_parser.add_argument("--ADNI_nifti_dir", type=str, help="Path to ADNI NIfTI directory")

    # Subparser for simulating data
    # > py main.py simulate --path "D:\ADNI_NIfTIs\3T\ADNI_002_S_0413_MR_Double_TSE_br_raw_20061115141733_1_S22682_I30117.nii.gz" --axis 2 --slice 24 --compare True
    # > py main.py simulate --path "D:\data-brats-2024\BraSyn\train\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2f.nii.gz" --axis 2 --slice 65
    simulate_parser = subparsers.add_parser("simulate", help="Simulate data")
    simulate_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to simulate")
    simulate_parser.add_argument("--axis", type=int, default=0, help="Axis for simulation")
    simulate_parser.add_argument("--slice", type=int, default=24, help="Slice index for simulation")
    simulate_parser.add_argument("--compare", type=bool, default=False, help="Whether to compare with original data (only for ADNI)")

    # Subparser for analysing noise
    # > py main.py analyse-snr-avg --dataset_dir "D:\ADNI_NIfTIs" --axis 2
    # > py main.py analyse-snr-avg --dataset_dir "D:\data-brats-2024" --axis 2 --seq "t2f" --dataset "BraSyn"
    analyse_noise_parser = subparsers.add_parser("analyse-snr-avg", help="Analyse noise in data")
    analyse_noise_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_noise_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_noise_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_noise_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for analysing SNR map
    # > py main.py analyse-snr-map --dataset_dir "D:\ADNI_NIfTIs" --axis 2 --slice 24
    # > py main.py analyse-snr-map --dataset_dir "D:\data-brats-2024" --axis 2 --slice 65 --seq "t2f" --dataset "BraSyn"
    analyse_snr_map_parser = subparsers.add_parser("analyse-snr-map", help="Analyse SNR map in data")
    analyse_snr_map_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_snr_map_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_snr_map_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    analyse_snr_map_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_snr_map_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for analysing brightness
    # > py main.py analyse-brightness --dataset_dir "D:\ADNI_NIfTIs" --slice 24 --axis 2
    # > py main.py analyse-brightness --dataset_dir "D:\data-brats-2024" --slice 65 --axis 2 --seq "t2f" --dataset "BraSyn"
    analyse_brightness_parser = subparsers.add_parser("analyse-brightness", help="Analyse brightness in data")
    analyse_brightness_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_brightness_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_brightness_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    analyse_brightness_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_brightness_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for batch simulation
    # > py main.py batch-simulate --brats_dir "D:\data-brats-2024" --output_dir "D:\data-brats-2024_simulated"
    batch_convert_parser = subparsers.add_parser("batch-simulate", help="Simulate data in batch")
    batch_convert_parser.add_argument("--brats_dir", type=str, required=True, help="Path to BraTS directory")
    batch_convert_parser.add_argument("--output_dir", type=str, help="Output directory for converted data")

    # Subparser for viewing data
    # > py main.py view --path "D:\data-brats-2024\GoAT\train-WithGroundTruth\BraTS-GoAT-00000\BraTS-GoAT-00000-t2f.nii.gz" --slice 65 --axis 2
    view_parser = subparsers.add_parser("view", help="View NIfTI data")
    view_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to view")
    view_parser.add_argument("--slice", type=int, default=24, help="Slice index for viewing")
    view_parser.add_argument("--axis", type=int, default=0, help="Axis for viewing")
    
    # Subparser for segmenting data
    # py main.py segment
    segment_parser = subparsers.add_parser("segment", help="Segment NIfTI data into white matter, grey matter, and CSF")

    args = parser.parse_args()
    action = args.action.lower()

    # Construct ADNI directory if it doesn't exist
    if action == "convert-adni":
        convert_adni(args)

    # Apply degradation to slice in a volume
    elif action == "simulate":
        simulate(args)

    # Perform analysis between two types of scans
    elif action == "analyse-snr-avg" or action == "analyse-brightness" or action == "analyse-snr-map":
        analyse(args)

    # Apply degradation to BraTS scans
    elif action == "batch-simulate":
        batch_simulate(args)

    # View a slice of a NIfTI file
    elif action == "view":
        view(args)

    # Segment a NIfTI file into white matter, grey matter, and CSF
    elif action =="segment":
        segment()