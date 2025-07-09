import matplotlib.pyplot as plt
from src.utils.inference import load_model, run_model_on_slice
from src.utils.plot import plot_lr_sr_hr

def predict(args):
    # Load the model
    model = load_model(args.model_path, rrdb_count=args.rrdb_count)

    # Run the model on the specified slice
    sr_slice, hr_slice, lr_slice = run_model_on_slice(
        model,
        args.lmdb_path,
        args.vol_name,
        args.set,
        args.slice
    )

    fig = plot_lr_sr_hr(lr_slice, sr_slice, hr_slice)
    plt.show()