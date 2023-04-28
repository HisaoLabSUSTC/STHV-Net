import time

import torch
from torch.utils.data import DataLoader
from STHVNet import STHVNet
from MaskedModel import MaskedModel

from Loss import PCTLoss

from Utils.HVDataLoader import HVDataset

from tqdm import tqdm

# Model parameters
OBJECTIVE_NUM = 10

TRANS_OUT_NUM = 16
TRANS_OUT_DIM = 256
HIDDEN_DIM = 256
USE_SAB = True
USE_RES = True
FORWARD_LAYERS = 4

DEPTH = 4

# Training parameters
BATCH_SIZE = 100
DEVICE = 'cuda:0'
MODEL = 0

if __name__ == "__main__":
    print("Device Name: ", DEVICE)
    model_name = "STHV-Net-64-I.ckpt"
    model_path = "./models/M10/old/" + model_name

    splitted_name = model_name.split('-')

    # STHV-Net-X.ckpt
    if len(splitted_name) == 3:
        hidden_dim = int(splitted_name[-1].split('.')[0])
        TRANS_OUT_NUM = 16
        TRANS_OUT_DIM = hidden_dim
        HIDDEN_DIM = hidden_dim
        USE_SAB = True
        USE_RES = True
        FORWARD_LAYERS = 4
        MODEL = 0
    # STHV-Net-X-I.ckpt
    if len(splitted_name) == 4:
        hidden_dim = int(splitted_name[2])
        TRANS_OUT_NUM = 16
        TRANS_OUT_DIM = hidden_dim
        HIDDEN_DIM = hidden_dim
        USE_SAB = False
        USE_RES = True
        FORWARD_LAYERS = 4
        MODEL = 0
    # MaskedModel-X.ckpt
    if len(splitted_name) == 2:
        DEPTH = int(splitted_name[-1].split('.')[0])
        HIDDEN_DIM = 128
        MODEL = 1

    if MODEL == 0:
        approximator = STHVNet(
            transInputDim=OBJECTIVE_NUM,
            transOutputNum=TRANS_OUT_NUM,
            transOutputDim=TRANS_OUT_DIM,
            hiddenDim=HIDDEN_DIM,
            useSAB=USE_SAB,
            forwardLayers=FORWARD_LAYERS,
            resOn=True
        ).to(DEVICE)
    if MODEL == 1:
        approximator = MaskedModel(
            input_dim=OBJECTIVE_NUM,
            device=DEVICE,
            depth=DEPTH,
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)

    # deal with compatibility issues
    state_dict = torch.load(model_path, map_location='cuda:0')
    has_prefix = False
    for key in state_dict.keys():
        if key.startswith('_orig_mod.'):
            has_prefix = True
            break
    if has_prefix:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    approximator.load_state_dict(state_dict)

    # Prediction
    testSet = HVDataset(dataDir="../TACSD/Datasets/Short", objectNum=OBJECTIVE_NUM, seeds=[5])

    testLoader = DataLoader(
        dataset=testSet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    approximator.eval()
    pbar = tqdm(testLoader)
    criterion = PCTLoss()
    accumulate_loss = 0
    batch_num = 0
    start_time = time.time()
    for batch in pbar:
        batch_num += 1
        VS, HV = batch
        with torch.no_grad():
            results = approximator(VS.to(DEVICE))
        loss = criterion(results, HV.to(DEVICE))

        accumulate_loss += loss.item()

        pbar.set_postfix(
            loss=loss.item(),
            average_epoch_loss=accumulate_loss / batch_num,
            mode="testing"
        )
    print(time.time() - start_time)
