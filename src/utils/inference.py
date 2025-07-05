import torch
import lmdb
import pickle
import gzip
from src.train.models.ESRGAN import Generator
from src.train.models.FSRCNN import FSRCNN

def run_model_on_slice(model_path, lmdb_path, vol_name, set_type, slice_index, rrdb_count):
    if "esrgan" in model_path.lower():
        model = Generator(rrdb_count=rrdb_count).to("cuda")
    elif "fsrcnn" in model_path.lower():
        model = FSRCNN().to("cuda")

    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    hr_key = f"{set_type}/{vol_name}/HR/{slice_index:03d}".encode("utf-8")
    lr_key = f"{set_type}/{vol_name}/LR/{slice_index:03d}".encode("utf-8")

    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            hr_slice = pickle.loads(gzip.decompress(txn.get(hr_key)))
            lr_slice = pickle.loads(gzip.decompress(txn.get(lr_key)))

    lr_tensor = torch.tensor(lr_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_slice = sr_tensor.squeeze().cpu().numpy()
    hr_slice = hr_slice.squeeze()
    lr_slice = lr_slice.squeeze()

    return sr_slice, hr_slice, lr_slice