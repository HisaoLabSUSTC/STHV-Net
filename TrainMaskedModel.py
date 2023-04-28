import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


from Loss import PCTLoss, NormPCTLoss
from MaskedModel import MaskedModel

from Utils.HVDataLoader import getDataLoader, HVDataset

from tqdm import tqdm


# Function to ensure experiments replicable
def lock_random(luckySeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(luckySeed)
    torch.manual_seed(luckySeed)
    random.seed(luckySeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(luckySeed)
        torch.cuda.manual_seed_all(luckySeed)


# Model parameters
OBJECTIVE_NUM = 10
HIDDEN_DIM = 256
DEPTH = 4
COSINE_ANNEALING = True

# Training parameters
LUCKY_SEED = 3402
NUM_EPOCH = 126
TRAIN_PROPORTION = 0.9
LEARNING_RATE = 15*1e-5
BATCH_SIZE = 200
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    lock_random(LUCKY_SEED)

    trainDataLoader, validDataLoader = getDataLoader(
        batchSize=BATCH_SIZE,
        workerNum=4,
        objectNum=OBJECTIVE_NUM,
        dataDir="Datasets",
        trainProportion=TRAIN_PROPORTION,
        seeds=[3, 4]
    )

    print("Device Found, Named", DEVICE)
    # approximator = torch.compile(MaskedModel(input_dim=TARGET_NUM, depth=DEPTH, device=DEVICE)).to(DEVICE)
    approximator = MaskedModel(input_dim=OBJECTIVE_NUM, depth=DEPTH, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(approximator.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=2, T_mult=2
    )
    model_name = \
        f"./models/" \
        f"MBertVolume2_" \
        f"OBJECTIVE_NUM{OBJECTIVE_NUM}_"\
        f"HIDDEN_DIM{HIDDEN_DIM}" \
        f".ckpt"
    result_name = model_name + '.rst.txt'

    # Main training loop
    norm_weight = 0.1
    min_valid_loss = torch.inf
    for epoch in range(NUM_EPOCH):
        # Training
        criterion = NormPCTLoss(norm_weight)
        approximator.train()
        pbar = tqdm(trainDataLoader)
        accumulate_loss = 0
        batch_num = 0
        for batch in pbar:
            batch_num += 1
            VS, HV = batch
            VS, HV = VS.to(DEVICE), HV.to(DEVICE)
            MASK = (torch.sum(VS, dim=2, keepdim=True) != 0).int()
            MASK = torch.cat((torch.ones((BATCH_SIZE, 1, 1)).to(DEVICE), MASK), dim=1)
            MASK = MASK.bmm(MASK.transpose(1, 2))
            MASK[:, :, 0] = 0
            MASK[:, 0, 0] = 1
            results = approximator.forward(VS, MASK)
            loss = criterion(results, HV)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(approximator.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            # scheduler.step()
            accumulate_loss += loss.item()
            pbar.set_postfix(
                now_epoch=epoch,
                loss=loss.item(),
                average_epoch_loss=accumulate_loss / batch_num,
                mode="training",
                learning_rate=scheduler.get_last_lr()
            )
        with open(result_name, mode='a+') as result_file:
            result_file.write(
                f'now_epoch={epoch}, '
                f'loss={loss.item()}, '
                f'average_epoch_loss={accumulate_loss / batch_num}, '
                f'mode=training, '
                f'learning_rate={scheduler.get_last_lr()}\n'
            )
        if COSINE_ANNEALING:
            scheduler.step()

        # Validation
        approximator.eval()
        pbar = tqdm(validDataLoader)
        accumulate_loss = 0
        batch_num = 0
        criterion = PCTLoss()
        for batch in pbar:
            batch_num += 1
            VS, HV = batch
            VS, HV = VS.to(DEVICE), HV.to(DEVICE)
            MASK = (torch.sum(VS, dim=2, keepdim=True) != 0).int()
            MASK = torch.cat((torch.ones((BATCH_SIZE, 1, 1)).to(DEVICE), MASK), dim=1)
            MASK = MASK.bmm(MASK.transpose(1, 2))
            MASK[:, :, 0] = 0
            MASK[:, 0, 0] = 1
            results = approximator(VS, MASK)
            with torch.no_grad():
                loss = criterion(results, HV)

            accumulate_loss += loss.item()
            pbar.set_postfix(
                now_epoch=epoch,
                loss=loss.item(),
                average_epoch_loss=accumulate_loss / batch_num,
                mode="validation"
            )
        with open(result_name, mode='a+') as result_file:
            result_file.write(
                f'now_epoch={epoch}, '
                f'loss={loss.item()}, '
                f'average_epoch_loss={accumulate_loss / batch_num}, '
                f'mode=validation\n'
            )

        if accumulate_loss / batch_num < min_valid_loss:
            min_valid_loss = accumulate_loss / batch_num

            torch.save(
                approximator.state_dict(),
                model_name
            )
            print(f"Model in Epoch {epoch} Saved! With Average Validation Loss: {accumulate_loss / batch_num}")
            with open(result_name, mode='a+') as result_file:
                result_file.write(
                    f"Model in Epoch {epoch} Saved! With Average Validation Loss: {accumulate_loss / batch_num}\n"
                )

            time.sleep(0.1)

        norm_weight *= 0.8

    # Prediction
    testSet = HVDataset(dataDir="./Datasets", objectNum=OBJECTIVE_NUM, seeds=[5])

    testLoader = DataLoader(
        dataset=testSet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    approximator.load_state_dict(torch.load(model_name))
    approximator.eval()

    pbar = tqdm(testLoader)
    criterion = PCTLoss()
    accumulate_loss = 0
    batch_num = 0
    for batch in pbar:
        batch_num += 1
        VS, HV = batch
        VS, HV = VS.to(DEVICE), HV.to(DEVICE)
        MASK = (torch.sum(VS, dim=2, keepdim=True) != 0).int()
        MASK = torch.cat((torch.ones((BATCH_SIZE, 1, 1)).to(DEVICE), MASK), dim=1)
        MASK = MASK.bmm(MASK.transpose(1, 2))
        MASK[:, :, 0] = 0
        MASK[:, 0, 0] = 1
        with torch.no_grad():
            results = approximator(VS, MASK)

        loss = criterion(results, HV)

        accumulate_loss += loss.item()

        pbar.set_postfix(
            loss=loss.item(),
            average_epoch_loss=accumulate_loss / batch_num,
            mode="testing"
        )
    with open(result_name, mode='a+') as result_file:
        result_file.write(str(accumulate_loss / batch_num))
