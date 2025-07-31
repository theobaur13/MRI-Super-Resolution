import matplotlib.pyplot as plt
from src.utils.readwrite import read_nifti
from src.simulation.pipeline import simluation_pipeline

def simulate(args):
    nifti_3T = read_nifti(args.path)
    _ = simluation_pipeline(nifti_3T, args.axis, visualize=True, slice=args.slice)
    plt.show()