import matplotlib.pyplot as plt
from src.utils.readwrite import read_nifti
from src.display.plot import display_img

def view(args):
    nifti = read_nifti(args.path, normalise=False)
    display_img([nifti], slice=args.slice, axis=args.axis)
    plt.show()