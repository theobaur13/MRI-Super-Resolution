import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.plot import plot_training_log

output_dir = "E:\ESRGAN_RRDB3_triple"
log_type = "psnr"
pretrain_loss_file = os.path.join(output_dir, f"{log_type}.csv")
plot_training_log(pretrain_loss_file, output_file=os.path.join(output_dir, f"{log_type}.png"))