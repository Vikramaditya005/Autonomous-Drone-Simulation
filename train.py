# train.py (patched)
import os
import glob
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparams
IMG_H, IMG_W = 128, 128           # must match collect_data.py
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 30
VAL_SPLIT = 0.1
# On Windows using multiple workers often causes issues; set to 0 if you see problems.
NUM_WORKERS = 0
if os.name != "nt":
    NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# reproducibility (optional)
torch.manual_seed(0)
np.random.seed(0)

# ----- Dataset -----
class AirSimDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.data_files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}. Run collect_data.py first.")
        self.transform = transform

        # Preload index mapping (episode_file, time_index)
        self.index = []  # list of (file_idx, t)
        self.file_cache = {}  # optionally cache small episodes
        for fi, fn in enumerate(self.data_files):
            with np.load(fn) as dd:
                if "rgbs" not in dd:
                    raise RuntimeError(f"File {fn} missing 'rgbs' key.")
                Tsteps = dd["rgbs"].shape[0]
            for t in range(Tsteps):
                self.index.append((fi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        fn = self.data_files[fi]

        # cache per-file arrays to avoid repeated IO
        if fn not in self.file_cache:
            dd = np.load(fn)
            # convert to desired dtype
            rgbs = dd["rgbs"].astype(np.uint8)     # (T,H,W,3)
            poses = dd["poses"].astype(np.float32) # (T,3)
            vels = dd["vels"].astype(np.float32)
            goals = dd["goals"].astype(np.float32)
            actions = dd["actions"].astype(np.float32)
            self.file_cache[fn] = dict(rgbs=rgbs, poses=poses, vels=vels, goals=goals, actions=actions)
        else:
            d = self.file_cache[fn]
            rgbs = d["rgbs"]
            poses = d["poses"]
            vels = d["vels"]
            goals = d["goals"]
            actions = d["actions"]

        img = rgbs[t]      # H,W,3 uint8
        pose = poses[t]    # (3,)
        vel = vels[t]      # (3,)
        goal = goals[t]    # (3,)
        action = actions[t]# (3,) -> target (vx,vy,vz)

        # Transform image (to tensor, normalize)
        if self.transform:
            img_t = self.transform(img)  # C,H,W float tensor
        else:
            img_t = torch.from_numpy(img).permute(2,0,1).float().div(255.0)

        # Build state-goal vector: pos (3) + goal (3) + vel (3) -> 9 dims
        state_goal = np.concatenate([pose, goal, vel]).astype(np.float32)  # (9,)
        state_goal_t = torch.from_numpy(state_goal).float()

        action_t = torch.from_numpy(action).float()

        return img_t, state_goal_t, action_t

# ----- Model -----
class PerceptionNet(nn.Module):
    def __init__(self, state_goal_dim=9, hidden=256, use_pretrained=False):
        super().__init__()

        # ResNet18 backbone
        backbone = models.resnet18(pretrained=use_pretrained)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Fully connected layers (exactly as in training)
        self.fc1 = nn.Linear(self.feature_dim + state_goal_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 3)  # vx, vy, vz

    def forward(self, img, state_goal):
        feat = self.backbone(img)              # B x 512
        x = torch.cat([feat, state_goal], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

# ----- Utilities -----
def collate_fn(batch):
    imgs, sgs, acts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    sgs = torch.stack(sgs, dim=0)
    acts = torch.stack(acts, dim=0)
    return imgs, sgs, acts

def train():
    # transforms
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),               # scales to [0,1], shape C,H,W
        # normalize with ImageNet stats (ok for resnet)
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # dataset and split
    dataset = AirSimDataset(DATA_DIR, transform=transform)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    pin_memory = True if DEVICE.type == "cuda" else False

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin_memory)

    model = PerceptionNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, sgs, acts in tqdm(train_loader, desc=f"Train E{epoch}"):
            imgs = imgs.to(DEVICE)
            sgs = sgs.to(DEVICE)
            acts = acts.to(DEVICE)

            preds = model(imgs, sgs)
            loss = criterion(preds, acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, sgs, acts in val_loader:
                imgs = imgs.to(DEVICE)
                sgs = sgs.to(DEVICE)
                acts = acts.to(DEVICE)
                preds = model(imgs, sgs)
                loss = criterion(preds, acts)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{EPOCHS}  TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f}")

        # save checkpoint
        ckpt_path = os.path.join(MODELS_DIR, f"ckpt_epoch_{epoch}.pth")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(MODELS_DIR, "ckpt_best.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, best_path)
            print("Saved new best:", best_path)

    print("Training finished. Best val loss:", best_val)

if __name__ == "__main__":
    train()
