# model.py
import torch, torch.nn as nn
import torchvision.models as models

class PerceptionNet(nn.Module):
    def __init__(self, img_size=(3,128,128), hidden=256, use_lstm=False):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        feat_dim = 512
        self.fc1 = nn.Linear(feat_dim + 6, hidden)  # 6 = state(3)+goal(3)
        self.relu = nn.ReLU()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 3) # vx,vy,vz

    def forward(self, img, state_goal, hx=None):
        # img: Bx3xHxW, state_goal: Bx6
        f = self.backbone(img)             # B x 512
        x = torch.cat([f, state_goal], dim=1)
        x = self.relu(self.fc1(x))         # B x hidden
        if self.use_lstm:
            x, hx = self.lstm(x.unsqueeze(1), hx)  # B x 1 x hidden
            x = x.squeeze(1)
        out = self.out(x)
        if self.use_lstm:
            return out, hx
        return out
