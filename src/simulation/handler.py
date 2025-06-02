import matplotlib.pyplot as plt
from src.utils.readwrite import read_nifti
from src.simulation.pipeline import simluation_pipeline
from src.display.plot import display_img
from src.utils.paths import get_matching_adni_scan

def simulate(args):
    # Arguments
    axis = args.axis
    slice_idx = args.slice
    path = args.path
    compare = args.compare

    if compare:
        # Get the matching 1.5T and 3T scans
        path_1_5T, path_3T = get_matching_adni_scan(path)
        nifti_1_5T = read_nifti(path_1_5T)
    else:
        path_3T = path
    
    nifti_3T = read_nifti(path_3T)

    simulated_nifti = simluation_pipeline(nifti_3T, axis, path, visualize=True, slice=slice_idx)

    if compare:
        # Display the target vs simulated image
        display_img([nifti_1_5T, simulated_nifti], slice=slice_idx, axis=axis, titles=["Original 1.5T Image", "Simulated 1.5T Image"])
    else:
        # Display the simulated image
        display_img([nifti_3T, simulated_nifti], slice=slice_idx, axis=axis, titles=["Original 3T Image", "Simulated 1.5T Image"])    
    plt.show()