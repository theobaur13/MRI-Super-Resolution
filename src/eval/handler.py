from src.eval.matter import matter
from src.eval.mae import mae
from src.eval.ssim import ssim
from src.eval.psnr import psnr
from src.eval.lpips import LPIPS
from src.eval.tumor import tumor
from src.eval.metrics import metrics
from src.eval.slice_eval import slice_eval

def evaluate(args):
    method = args.method.lower()

    if method == "matter":
        matter(args.model_path, args.lmdb_path, args.flywheel_dir, args.working_dir, args.set_type)
    elif method == "tumor":
        tumor(args.model_path, args.deepseg_path, args.lmdb_path, args.working_dir, args.set_type)
    elif method == "metrics":
        metrics(args.model_path, args.lmdb_path, args.set_type)
    elif method == "mae":
        mae(args.model_path, args.lmdb_path, args.set_type)
    elif method == "ssim":
        ssim(args.model_path, args.lmdb_path, args.set_type)
    elif method == "psnr":
        psnr(args.model_path, args.lmdb_path, args.set_type)
    elif method == "lpips":
        LPIPS(args.model_path, args.lmdb_path, args.set_type)
    elif method == "slice":
        slice_eval(args.model_path, args.lmdb_path, args.set_type)