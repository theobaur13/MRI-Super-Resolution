# MRI-Super-Resolution
Deep-learning based super-resolution can enhance 1.5T images to resemble 3T-like quality, aiming to match the diagnostic benefits of higher-quality images without access to a 3T machine. 

This project degrades a set of 3T images from the BraTS 2024 dataset to resemble 1.5T using $k$-space filtering and simulate GRAPPA and Partial Fourier acquisition artefacts. Using this data, we train an ESRGAN with a composite weighted loss function:
- VGG19 perceptual loss, L1 pixel loss, and Sobel edge loss
- VGG19 perceptual loss, L1 pixel loss, and Fourier loss 

To assess downstream utility, we processed the enhanced images from both
models and non-enhanced low quality images using FSL-FAST (for grey matter,
white matter, and cerebrospinal fluid segmentation) and DeepSeg (for tumour
segmentation). In both cases, the enhanced images exhibited performance more
closely aligned with high-quality images than with low-quality counterparts,
suggesting that the super-resolution provides images that contain clinically
useful information.

## Prerequisites
This project was conducted with Python 3.10.0 using CUDA 12.9 on Windows. We also use Docker to run FSL-FAST during the evaluation.

## Installation
Clone repository:
```bash
git clone https://github.com/theobaur13/MRI-Super-Resolution
```

Set up virtual environment:
```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Clone [BraTS 2024](https://www.synapse.org/Synapse:syn53708249/wiki/626323) dataset. Dataset must have the format:
```
data-brats-2024/
├── BraSyn
│   ├── train
│   │   ├── BraTS-GLI-00000-000
│   │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   │   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   │   │   └── BraTS-GLI-00000-000-t2w.nii.gz
│   │   └── ...
│   └── validate
│       ├── BraTS-GLI-00000-000
│       │   ├── BraTS-GLI-00001-000-t1c.nii.gz
│       │   ├── BraTS-GLI-00001-000-t1n.nii.gz
│       │   ├── BraTS-GLI-00001-000-t2f.nii.gz
│       │   └── BraTS-GLI-00001-000-t2w.nii.gz
│       └── ...
├── GLI
├── GoAT
├── LocalInpainting
├── MEN-RT
├── MET
├── Path
└── PED
```

Create a `.env` file inside the root directory and fill with the following information (note that LMDB will not exist until created during preprocessing):
```
LMDB_PATH="/path/to/LMDB"
BRATS_DIR="/path/to/data-brats-2024"
FLYWHEEL_DIR="/path/to/MRI-Super-Resolution/flywheel"
PROJECT_ROOT="/path/to/MRI-Super-Resolution"
DEEPSEG_PATH="/path/to/MRI-Super-Resolution/src/eval/DeepSeg/UNet_UNet.hdf5"
```

## Preprocessing
To generate training data from the BraTS dataset, create an LMDB using:
```
py main.py generate-training-data 
--output_dir "/path/to/LMDB"
--seq "t2f"
--limit None
--axis 2
--normalise True
```
The LMDB will be found at `output_dir`, which should also be filled into the `LMDB_PATH` variable in `.env`.

## Training
Using the LMDB, the ESRGAN and FSRCNN models can be trained using:
```
py main.py train
--output_dir "/path/to/model"
--resume False
```
If resuming from a previous checklist then `resume` can be set to `True`.

## Evaluation
To run the trained model on a single slice and display a super-resoloved image, the following command can be used. `rrdb_count` represents the number of RRDBs used in the ESRGAN model trained (default 3).
```
py main.py predict
--model_path "/path/to/model"
--vol_name "BraTS-GLI-00000-000-t2f"
--set ["train", "validate", "test"]
--slice 0-155
--rrdb_count 3
```

To visualise MAE differences between the original high-quality image and the super-resolved image, a MAE map can be displayed using:
```
py main.py error-map
--model_path "/path/to/model"
--vol_name "BraTS-GLI-00000-000-t2f"
--set ["train", "validate", "test"]
--slice 0-155
--rrdb_count 3
```

To generate `.png` images of all super-resolved slices using the trained model use:
```
py main.py export-predictions
--model_path "/path/to/model"
--output_dir "/path/to/output_dir"
--rrdb_count 3
--set_type ["train", "validate", "test"]
```

To perform statistical analysis on the trained model's output, use the following command. `matter` uses FSL-FAST for GM, WM, and CSF segmentation and `tumor` uses DeepSeg for tumour segmentation, where segmentation maps by the trained model are compared to segmenatation maps from the degraded and original high-quality scans. `metrics` simply performs SSIM, PSNR, and LPIPS in one pass.
```
py main.py evaluate
--model_path "/path/to/model"
--working_dir "/path/to/working_dir"
--set_type ["train", "validate", "test"]
--method ["matter", "mae", "ssim", "psnr", "lpips", "tumor", "metrics", "slice"]
```

## Utilities
To view a single MRI slice from a volume, use:
```
py main.py view
--path "/path/to/volume"
--slice 65
--axis 2
```
To view a slice when degradation has been applied, use:
```
py main.py simulate
--path "/path/to/volume"
--slice 65
--axis 2
```