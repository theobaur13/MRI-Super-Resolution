import os
import argparse
from src.analysis.handler import analyse
from src.display.handler import view
from src.simulation.handler import simulate
from src.train.handler import train
from src.segment.handler import segment
from src.convert_adni.handler import convert_adni
from src.generate_training_data.handler import generate_training_data

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Subparser for converting ADNI data
    # > py main.py convert-adni --ADNI_dir "E:\ADNI" --ADNI_nifti_dir "D:\ADNI_NIfTIs"
    convert_parser = subparsers.add_parser("convert-adni", help="Convert ADNI data to NIfTI format")
    convert_parser.add_argument("--ADNI_dir", type=str, help="Path to ADNI directory")
    convert_parser.add_argument("--ADNI_nifti_dir", type=str, help="Path to ADNI NIfTI directory")

    # Subparser for simulating data
    # > py main.py simulate --path "E:\ADNI_NIfTIs\3T\ADNI_002_S_0413_MR_Double_TSE_br_raw_20061115141733_1_S22682_I30117.nii.gz" --axis 2 --slice 24 --compare True
    # > py main.py simulate --path "E:\data-brats-2024\BraSyn\train\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz" --axis 2 --slice 65
    simulate_parser = subparsers.add_parser("simulate", help="Simulate data")
    simulate_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to simulate")
    simulate_parser.add_argument("--axis", type=int, default=0, help="Axis for simulation")
    simulate_parser.add_argument("--slice", type=int, default=24, help="Slice index for simulation")
    simulate_parser.add_argument("--compare", type=bool, default=False, help="Whether to compare with original data (only for ADNI)")

    # Subparser for analysing noise
    # > py main.py analyse-snr-avg --dataset_dir "E:\ADNI_NIfTIs" --axis 2
    # > py main.py analyse-snr-avg --dataset_dir "E:\data-brats-2024" --axis 2 --seq "t2f" --dataset "BraSyn"
    analyse_noise_parser = subparsers.add_parser("analyse-snr-avg", help="Analyse noise in data")
    analyse_noise_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_noise_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_noise_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_noise_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for analysing SNR map
    # > py main.py analyse-snr-map --dataset_dir "E:\ADNI_NIfTIs" --axis 2 --slice 24
    # > py main.py analyse-snr-map --dataset_dir "E:\data-brats-2024" --axis 2 --slice 65 --seq "t2f" --dataset "BraSyn"
    analyse_snr_map_parser = subparsers.add_parser("analyse-snr-map", help="Analyse SNR map in data")
    analyse_snr_map_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_snr_map_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_snr_map_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    analyse_snr_map_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_snr_map_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for analysing brightness
    # > py main.py analyse-brightness --dataset_dir "E:\ADNI_NIfTIs" --slice 24 --axis 2
    # > py main.py analyse-brightness --dataset_dir "E:\data-brats-2024" --slice 65 --axis 2 --seq "t2f" --dataset "BraSyn"
    analyse_brightness_parser = subparsers.add_parser("analyse-brightness", help="Analyse brightness in data")
    analyse_brightness_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    analyse_brightness_parser.add_argument("--axis", type=int, default=0, help="Axis for analysis")
    analyse_brightness_parser.add_argument("--slice", type=int, default=24, help="Slice index for analysis")
    analyse_brightness_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")
    analyse_brightness_parser.add_argument("--dataset", type=str, required=False, help="Dataset for conversion (e.g., 'BraSyn', 'GLI')")

    # Subparser for generating training data
    # > py main.py generate-training-data --brats_dir "E:\data-brats-2024" --output_dir "E:\data-brats-2024_simulated" --axis 2 --limit 100
    generate_training_data_parser = subparsers.add_parser("generate-training-data", help="Generate simulated training data for BraTS")
    generate_training_data_parser.add_argument("--brats_dir", type=str, required=True, help="Path to BraTS directory")
    generate_training_data_parser.add_argument("--output_dir", type=str, help="Output directory for converted data")
    generate_training_data_parser.add_argument("--axis", type=int, required=True, help="Axis for simulation")
    generate_training_data_parser.add_argument("--limit", type=int, help="Limit the number of files to simulate")
    generate_training_data_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")

    # Subparser for viewing data
    # > py main.py view --path "E:\data-brats-2024\BraSyn\train\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2w.nii.gz" --slice 65 --axis 2
    view_parser = subparsers.add_parser("view", help="View NIfTI data")
    view_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to view")
    view_parser.add_argument("--slice", type=int, default=24, help="Slice index for viewing")
    view_parser.add_argument("--axis", type=int, default=0, help="Axis for viewing")
    
    # Subparser for segmenting data
    # py main.py segment --dataset_dir "E:\data-brats-2024" --limit 1
    segment_parser = subparsers.add_parser("segment", help="Segment NIfTI data into white matter, grey matter, and CSF")
    segment_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    segment_parser.add_argument("--limit", type=int, default=1, help="Limit the number of files to segment")

    # Subparser for training a model
    # py main.py train --dataset_dir "E:\data-brats-2024_simulated"
    training_parser = subparsers.add_parser("train", help="Train a model on the dataset")
    training_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")

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
    elif action == "generate-training-data":
        generate_training_data(args)

    # View a slice of a NIfTI file
    elif action == "view":
        view(args)

    # Segment a NIfTI file into white matter, grey matter, and CSF
    elif action =="segment":
        segment(args)

    # Train a model on the dataset
    elif action == "train":
        train(args)