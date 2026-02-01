# evaluate_perception.py
import os, glob, math, argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
from train import AirSimDataset, PerceptionNet, IMG_H, IMG_W, DATA_DIR, MODELS_DIR  # reuse train.py definitions

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BATCH_SIZE = 32
NUM_WORKERS = 0
CKPT = os.path.join(MODELS_DIR, "ckpt_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrics_for_arrays(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    return mae, mse, rmse, r2

def main():
    print("Device:", DEVICE)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # load dataset and split (same split as training script - use VAL_SPLIT=0.1)
    dataset = AirSimDataset(DATA_DIR, transform=transform)
    # use same split fraction as train.py
    VAL_SPLIT = 0.1
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda b: zip(*b) )

    # build model and load checkpoint
    model = PerceptionNet().to(DEVICE)
    if not os.path.isfile(CKPT):
        print("Checkpoint not found:", CKPT)
        return
    chk = torch.load(CKPT, map_location=DEVICE)
    # extract state dict from standard saved checkpoint
    state_dict = None
    if isinstance(chk, dict):
        for k in ("model_state", "state_dict", "model", "model_state_dict"):
            if k in chk:
                state_dict = chk[k]
                break
        if state_dict is None and any("." in str(k) for k in chk.keys()):
            state_dict = chk
    else:
        state_dict = chk

    if state_dict is None:
        print("Couldn't locate model_state in checkpoint. Aborting.")
        return

    try:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Loaded model.")
    except Exception as e:
        print("Error loading model:", e)
        return

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in val_loader:
            imgs, sgs, acts = batch
            imgs_t = torch.stack(list(imgs), dim=0).to(DEVICE)
            sgs_t = torch.stack(list(sgs), dim=0).to(DEVICE)
            acts_t = torch.stack(list(acts), dim=0).to(DEVICE)

            preds = model(imgs_t, sgs_t)
            preds_np = preds.cpu().numpy()
            acts_np = acts_t.cpu().numpy()

            all_true.append(acts_np)
            all_pred.append(preds_np)

    if not all_true:
        print("No validation samples found.")
        return

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    mae, mse, rmse, r2 = metrics_for_arrays(y_true, y_pred)

    D = y_true.shape[1]
    print("Evaluation results on validation set:")
    for d in range(D):
        print(f" Dim {d}: MAE={mae[d]:.6f}  RMSE={rmse[d]:.6f}  R2={r2[d]:.6f}")
    print(" Overall MAE (avg dims): {:.6f}".format(np.mean(mae)))
    print(" Overall RMSE (avg dims): {:.6f}".format(np.mean(rmse)))
    print(" Overall R2 (avg dims): {:.6f}".format(np.mean(r2)))

if __name__ == "__main__":
    main()
