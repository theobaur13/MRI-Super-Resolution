import matplotlib.pyplot as plt
from src.utils.inference import load_model, run_model_on_slice
from src.utils.plot import plot_lr_sr_hr

def error_map(args):
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

    plot_lr_sr_hr(lr_slice, sr_slice, hr_slice)

    # Calculate the MAE map
    mae = (hr_slice - sr_slice) ** 2

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(mae, cmap='plasma', vmin=0, vmax=mae.max())
    plt.title('Map')
    plt.colorbar(label='Error')

    plt.tight_layout()
    plt.show()