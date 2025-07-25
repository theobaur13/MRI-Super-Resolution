import matplotlib.pyplot as plt
from src.utils.readwrite import read_nifti
from src.simulation.pipeline import simluation_pipeline
from src.utils.plot import display_img

def simulate(args):
    nifti_3T = read_nifti(args.path)
    simulated_nifti = simluation_pipeline(nifti_3T, args.axis, visualize=True, slice=args.slice)
    plt.show()