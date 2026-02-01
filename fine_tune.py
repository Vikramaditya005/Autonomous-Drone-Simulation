# fine_tune.py (place next to train.py)
import os, torch
from train import PerceptionNet, DATA_DIR, MODELS_DIR, DEVICE, collate_fn, BATCH_SIZE, NUM_WORKERS, LR
from train import AirSimDataset   # uses your train.py dataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# load dataset (only new data if you prefer)
dataset = AirSimDataset(DATA_DIR, transform=None)  # or use same transform as train.py
n_val = max(1, int(len(dataset) * 0.1))
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

model = PerceptionNet().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR * 0.1)  # lower LR for fine-tuning

# load checkpoint
ckpt = torch.load(os.path.join(MODELS_DIR, "ckpt_best.pth"), map_location=DEVICE)
if "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)

# few epochs
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    for imgs, sgs, acts in train_loader:
        imgs = imgs.to(DEVICE); sgs = sgs.to(DEVICE); acts = acts.to(DEVICE)
        preds = model(imgs, sgs)
        loss = torch.nn.functional.mse_loss(preds, acts)
        opt.zero_grad(); loss.backward(); opt.step()
    print("Fine-tune epoch", epoch, "done")

torch.save({"epoch": 0, "model_state": model.state_dict()}, os.path.join(MODELS_DIR, "ckpt_finetuned.pth"))
print("Saved finetuned model.")
