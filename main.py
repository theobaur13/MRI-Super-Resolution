import os
import argparse
from dotenv import load_dotenv
from src.view import view
from src.simulation.handler import simulate
from src.train.handler import train
from src.eval.handler import evaluate
from src.generate_training_data import generate_training_data
from src.predict import predict
from src.error_map import error_map
from src.export_predictions import export_predictions

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    LMDB_PATH = os.getenv("LMDB_PATH")
    BRATS_DIR= os.getenv("BRATS_DIR")
    FLYWHEEL_DIR = os.getenv("FLYWHEEL_DIR")

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
    # > py main.py generate-training-data --output_dir "E:\data-brats-2024_simulated" --axis 2 --limit 100 --seq "t2f"
    generate_training_data_parser = subparsers.add_parser("generate-training-data", help="Generate simulated training data for BraTS")
    generate_training_data_parser.add_argument("--brats_dir", type=str, default=BRATS_DIR, help="Path to BraTS dataset directory")
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

    # Subparser for training a model
    # py main.py train --output_dir "E:\models" --resume True
    training_parser = subparsers.add_parser("train", help="Train a model on the dataset")
    training_parser.add_argument("--lmdb_path", type=str, default=LMDB_PATH, help="Path to LMDB dataset")
    training_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model and logs")
    training_parser.add_argument("--resume", type=bool, default=False, help="Whether to resume training from a checkpoint")

    # Subparser for running a model on a slice
    # py main.py predict --model_path "E:\models\best_generator.pth" --vol_name "BraTS-GLI-00000-000-t2f" --set "train" --slice 24
    predict_parser = subparsers.add_parser("predict", help="Run a model on a slice")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    predict_parser.add_argument("--lmdb_path", type=str, default=LMDB_PATH, help="Path to LMDB dataset")
    predict_parser.add_argument("--vol_name", type=str, help="Volume name in LMDB dataset (e.g., BraTS-GLI-00000-000-t2f)")
    predict_parser.add_argument("--set", type=str, choices=["train", "validate"], default="train", help="Dataset set to run the model on")
    predict_parser.add_argument("--slice", type=int, default=24, help="Slice index for running the model")
    predict_parser.add_argument("--rrdb_count", type=int, default=3, help="Number of RRDB blocks in the generator")

    # Subparser for error map on a slice
    # py main.py error-map --model_path "E:\models\best_generator.pth" --vol_name "BraTS-GLI-00000-000-t2f" --set "train" --slice 24
    error_map_parser = subparsers.add_parser("error-map", help="Calculate error map for a model")
    error_map_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    error_map_parser.add_argument("--lmdb_path", type=str, default=LMDB_PATH, help="Path to LMDB dataset")
    error_map_parser.add_argument("--vol_name", type=str, help="Volume name in LMDB dataset (e.g., BraTS-GLI-00000-000-t2f)")
    error_map_parser.add_argument("--set", type=str, choices=["train", "validate"], default="train", help="Dataset set to calculate error map on")
    error_map_parser.add_argument("--slice", type=int, default=24, help="Slice index for running the model")
    error_map_parser.add_argument("--rrdb_count", type=int, default=3, help="Number of RRDB blocks in the generator")

    # Subparser for running model on dataset and exporting results
    # py main.py export-predictions --model_path "E:\models\best_generator.pth" --output_dir "E:\predictions"
    export_predictions_parser = subparsers.add_parser("export-predictions", help="Run model on dataset and export results")
    export_predictions_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    export_predictions_parser.add_argument("--lmdb_path", type=str, default=LMDB_PATH, help="Path to LMDB dataset")
    export_predictions_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predictions")
    export_predictions_parser.add_argument("--rrdb_count", type=int, default=3, help="Number of RRDB blocks in the generator")

    # Subparser for evaluating a model using various methods
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model using various methods")
    evaluate_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    evaluate_parser.add_argument("--lmdb_path", type=str, default=LMDB_PATH, help="Path to LMDB dataset")
    evaluate_parser.add_argument("--flywheel_dir", type=str, default=FLYWHEEL_DIR, help="Directory for Flywheel output")
    evaluate_parser.add_argument("--working_dir", type=str, help="Output directory for segmentation results")
    evaluate_parser.add_argument("--method", type=str, choices=["matter", "mae", "ssim", "psnr"], required=True, help="Evaluation method to use")

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

    # Train a model on the dataset
    elif action == "train":
        train(args)

    # Run a model on a slice
    elif action == "predict":
        predict(args)

    # Calculate loss map for a model
    elif action == "error-map":
        error_map(args)

    # Run model on dataset and export results
    elif action == "export-predictions":
        export_predictions(args)

    # Evaluate a model using various methods
    elif action == "evaluate":
        evaluate(args)