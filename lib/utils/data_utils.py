import torch
import os
import numpy as np

def create_time_series_dataloader(datapath="/data", batchsize=8, num_workers=4,logger=None):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        timex_file = os.path.join(datapath, "%s_x_original.npy" % split)
        timey_file = os.path.join(datapath, "%s_y_original.npy" % split)

        if not os.path.exists(timex_file) or not os.path.exists(timey_file):
            if logger:
                logger.warning(f"Data for split '{split}' not found in {datapath}. Skipping.")
            else:
                print(f"Warning: Data for split '{split}' not found in {datapath}. Skipping.")
            continue

        timex = np.load(timex_file)
        timex = torch.from_numpy(timex).to(dtype=torch.float32)
        timey = np.load(timey_file)
        timey = torch.from_numpy(timey).to(dtype=torch.float32)
        
        seq_info = os.path.basename(datapath)
        parent_dir = os.path.dirname(datapath)
        data_type = os.path.basename(parent_dir)
        if logger:
            logger.info("[Dataset: %s][Input_Output: %s][%s] Loaded %d examples." % (data_type, seq_info, split, timex.shape[0]))
        else:
            print("[Dataset: %s][Input_Output: %s][%s] Loaded %d examples." % (data_type, seq_info, split, timex.shape[0]))

        dataset = torch.utils.data.TensorDataset(timex, timey)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True if split == "train" else False,
            num_workers=num_workers,
            drop_last=True if split == "train" else False,
        )
    return dataloaders

def get_params(data_type):
   
    params_map = {
        "weather": {"batchsize": 128, "Sin": 21, "Sout": 21},
        "traffic": {"batchsize": 16, "Sin": 862, "Sout": 862},
        "electricity": {"batchsize": 32, "Sin": 321, "Sout": 321},
        "ETTh1": {"batchsize": 128, "Sin": 7, "Sout": 7},
        "ETTh2": {"batchsize": 128, "Sin": 7, "Sout": 7},
        "ETTm1": {"batchsize": 128, "Sin": 7, "Sout": 7},
        "ETTm2": {"batchsize": 128, "Sin": 7, "Sout": 7},
        "exchange": {"batchsize": 128, "Sin": 8, "Sout": 8},
    }
    if data_type not in params_map:
        raise ValueError(f"Unknown data type {data_type}")
    
    result = params_map[data_type]
  
    return result

def loss_fn(type, beta=1.0):
    if type == "mse":
        return torch.nn.MSELoss()
    elif type == "smoothl1":  # combine L1 and L2 loss
        return torch.nn.SmoothL1Loss(beta=beta)
    else:
        raise ValueError("Invalid type")
