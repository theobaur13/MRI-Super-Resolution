import matplotlib.pyplot as plt
from src.utils.inference import run_model_on_slice
from src.utils.plot import plot_lr_sr_hr

def predict(args):
    # Run the model on the specified slice
    sr_slice, hr_slice, lr_slice = run_model_on_slice(
        args.model_path,
        args.lmdb_path,
        args.vol_name,
        args.set,
        args.slice,
        args.rrdb_count
    )

    fig = plot_lr_sr_hr(lr_slice, sr_slice, hr_slice)
    plt.show()