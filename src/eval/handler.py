from src.eval.matter import matter
from src.eval.mae import mae

def evaluate(args):
    method = args.method.lower()

    if method == "matter":
        matter(args.model_path, args.lmdb_path, args.flywheel_dir, args.working_dir)
    elif method == "mae":
        mae(args.model_path, args.lmdb_path)