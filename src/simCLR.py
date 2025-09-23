import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchvision.datasets import ImageFolder

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from torch.optim.lr_scheduler import LambdaLR

# ----------------------------
# Config
# ----------------------------
DATA_ROOT = "data/eurosat_custom/train"   # prepared split (train only, unlabeled)
BATCH_SIZE = 256
EPOCHS = 150
LR = 0.06
NUM_WORKERS = 8
IMG_SIZE = 224             # resize inside transform
OUT_DIR = Path("checkpoints_ssl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
warmup_epochs = 10
total_epochs = EPOCHS

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    return 0.5 * (1.0 + torch.cos(torch.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = torch.cuda.is_available()  # mixed precision if GPU


# ----------------------------
# Model: ResNet18 encoder + SimCLR projection head
# ----------------------------
class SimCLR(nn.Module):
    def __init__(self, backbone, in_dim=512, proj_hidden=512, proj_out=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(in_dim, proj_hidden, proj_out)

    def forward(self, x):
        # backbone assumed to output [N, C, 1, 1] after global pooling
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

# Build a torchvision resnet18 backbone without the FC layer
resnet = tvm.resnet18(weights=None)
# replace avgpool+fc stack with Identity + keep global avgpool:
# torchvision resnet18 returns features after avgpool as 512-d before fc.
backbone = nn.Sequential(*list(resnet.children())[:-1])  # until avgpool, outputs [N,512,1,1]
model = SimCLR(backbone, in_dim=512, proj_hidden=512, proj_out=128).to(device)


# ----------------------------
# Data: EuroSAT train images as unlabeled pairs of views
# ----------------------------
# SimCLR default normalization in Lightly is ImageNet stats; perfect for ResNet18 at 224.
transform = SimCLRTransform(
    input_size=IMG_SIZE,
    gaussian_blur=0.1,        # EuroSAT is small; mild blur helps but keep modest
    cj_strength=0.5,          # color jitter strength
)

dataset = ImageFolder(DATA_ROOT, transform=transform)
# ImageFolder returns ( (v1, v2), label ) because transform yields two views.
# We'll ignore labels during pretraining.
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ----------------------------
# Objective & Optimizer
# ----------------------------
criterion = NTXentLoss(temperature=0.5)   # standard SimCLR temperature
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# ----------------------------
# Training loop
# ----------------------------
print(f"Starting SimCLR pretraining on {device} for {EPOCHS} epochs…")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for (v1, v2), _ in loader:      # labels are unused
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item()

    avg_loss = total_loss / len(loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch:03d} | loss {avg_loss:.5f} | lr {current_lr:.5f}")

    scheduler.step()

    # (optional) save checkpoints every N epochs
    if epoch % 25 == 0 or epoch == EPOCHS:
        # save only the encoder (backbone) weights for fine-tuning
        enc_state = {k.replace("backbone.", "", 1): v
                     for k, v in model.state_dict().items()
                     if k.startswith("backbone.")}
        torch.save(enc_state, OUT_DIR / f"simclr_resnet18_eurosat_epoch{epoch}.pt")

print("Done.")