import os
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from src.utils import *
from src.simulation import *

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    # Command line arguments for dataset type
    if len(sys.argv) < 2:
        print("Usage: python main.py [dataset_type]")
        print("dataset_type: brain or prostate")
        sys.exit(1)
    dataset_type = sys.argv[1].lower()

    print("Collecting paths...")
    if dataset_type == "brats":
        seq = "t2f"                                 # t1c, t1n, t2f, t2w
        dataset = "BraSyn"                             # BraSyn, GLI
        
        data_path = os.path.join(data_path, "data-brats-2024-master-BraSyn-train-BraTS-GLI-00000-000")
        paths, validate_paths = get_brats_paths(data_path, seq, dataset)

    elif dataset_type == "pccai":
        seq = "sag"                                 # adc, cor, hbv, sag, t2w
        fold = 0

        data_path = os.path.join(data_path, "data-picai-main")
        paths = get_picai_paths(data_path, fold, seq)

    elif dataset_type == "ixi":
        data_path = os.path.join(data_path, "IXI-T2")
        t1_5_paths, t3_paths = get_ixi_paths(data_path)
        paths = t1_5_paths + t3_paths

    t1_5_vs_t3(t1_5_paths, t3_paths, axis=1)
    generate_simulated_images(paths, dataset_type, axis=1)