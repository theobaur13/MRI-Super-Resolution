import os
import argparse
from src.display.handler import view
from src.simulation.handler import simulate
from src.train.handler import train
from src.segment.handler import segment
from src.generate_training_data.handler import generate_training_data
from src.run.handler import run_model
from src.error_map.handler import error_map

if __name__ == "__main__":
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Subparser for simulating data
    # > py main.py simulate --path "E:\data-brats-2024\BraSyn\train\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2f.nii.gz" --axis 2 --slice 65
    simulate_parser = subparsers.add_parser("simulate", help="Simulate data")
    simulate_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to simulate")
    simulate_parser.add_argument("--axis", type=int, default=2, help="Axis for simulation")
    simulate_parser.add_argument("--slice", type=int, default=24, help="Slice index for simulation")

    # Subparser for generating training data
    # > py main.py generate-training-data --brats_dir "E:\data-brats-2024" --output_dir "E:\data-brats-2024_simulated" --axis 2 --limit 100 --seq "t2f"
    generate_training_data_parser = subparsers.add_parser("generate-training-data", help="Generate simulated training data for BraTS")
    generate_training_data_parser.add_argument("--brats_dir", type=str, required=True, help="Path to BraTS directory")
    generate_training_data_parser.add_argument("--output_dir", type=str, help="Output directory for converted data")
    generate_training_data_parser.add_argument("--axis", type=int, default=2, help="Axis for simulation")
    generate_training_data_parser.add_argument("--limit", type=int, help="Limit the number of files to simulate")
    generate_training_data_parser.add_argument("--seq", type=str, required=False, help="Sequence type (e.g., 't1c', 't1n', 't2f', 't2w')")

    # Subparser for viewing data
    # > py main.py view --path "E:\data-brats-2024\BraSyn\train\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2f.nii.gz" --slice 65 --axis 2
    view_parser = subparsers.add_parser("view", help="View NIfTI data")
    view_parser.add_argument("--path", type=str, required=True, help="Path to NIfTI file to view")
    view_parser.add_argument("--slice", type=int, default=24, help="Slice index for viewing")
    view_parser.add_argument("--axis", type=int, default=2, help="Axis for viewing")
    
    # Subparser for segmenting data
    # py main.py segment --dataset_dir "E:\data-brats-2024" --limit 1
    segment_parser = subparsers.add_parser("segment", help="Segment NIfTI data into white matter, grey matter, and CSF")
    segment_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    segment_parser.add_argument("--limit", type=int, default=1, help="Limit the number of files to segment")

    # Subparser for training a model
    # py main.py train --dataset_dir "E:\data-brats-2024_simulated\train" --output_dir "E:\models" --axis 2
    training_parser = subparsers.add_parser("train", help="Train a model on the dataset")
    training_parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB dataset")
    training_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model and logs")
    training_parser.add_argument("--resume", type=bool, default=False, help="Whether to resume training from a checkpoint")

    # Subparser for running a model
    # py main.py run --model_path "E:\models\best_generator.pth" --lmdb_path "E:\data" --vol_name "BraTS-GLI-00000-000-t2f" --set "train" --slice 24
    run_parser = subparsers.add_parser("run", help="Run a model on a slice")
    run_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    run_parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB dataset")
    run_parser.add_argument("--vol_name", type=str, help="Volume name in LMDB dataset (e.g., BraTS-GLI-00000-000-t2f)")
    run_parser.add_argument("--set", type=str, choices=["train", "validate"], default="train", help="Dataset set to run the model on")
    run_parser.add_argument("--slice", type=int, default=24, help="Slice index for running the model")
    run_parser.add_argument("--rrdb_count", type=int, default=3, help="Number of RRDB blocks in the generator")

    # Subparser for error map
    # py main.py error --model_path "E:\models\best_generator.pth" --lmdb_path "E:\data" --vol_name "BraTS-GLI-00000-000-t2f" --set "train" --slice 24
    loss_parser = subparsers.add_parser("error", help="Calculate error map for a model")
    loss_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    loss_parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB dataset")
    loss_parser.add_argument("--vol_name", type=str, help="Volume name in LMDB dataset (e.g., BraTS-GLI-00000-000-t2f)")
    loss_parser.add_argument("--set", type=str, choices=["train", "validate"], default="train", help="Dataset set to calculate error map on")
    loss_parser.add_argument("--slice", type=int, default=24, help="Slice index for running the model")
    loss_parser.add_argument("--rrdb_count", type=int, default=3, help="Number of RRDB blocks in the generator")

    args = parser.parse_args()
    action = args.action.lower()

    # Apply degradation to slice in a volume
    if action == "simulate":
        simulate(args)

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

    # Run a model on a slice
    elif action == "run":
        run_model(args)

    # Calculate loss map for a model
    elif action == "error":
        error_map(args)