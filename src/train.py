import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.models as models
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ----------------- argparse -----------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train a small CNN on MNIST/Fashion-MNIST")
    p.add_argument(
        "--dataset", choices=["fashion-mnist", "mnist", "cifar10","eurosat"], default="fashion-mnist"
    )
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # legacy path args (we’ll map them into roots if provided)
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--metrics", type=str, default=None)
    p.add_argument("--reports-dir", type=str, default=None)
    # config
    p.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config with defaults",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["smallcnn", "resnet18_cifar", "resnet18_imagenet"],
        help="Choose model architecture",
    )
    return p


# ----------------- small utils -----------------
def get_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_cli_over_config_with_defaults(cfg, args, parser):
    cfg = deepcopy(cfg)
    defaults = parser.parse_args([])  # argparse defaults only
    for arg_name, cfg_key in [
        ("dataset", "dataset"),
        ("data_dir", "data_dir"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("num_workers", "num_workers"),
        ("seed", "seed"),
        ("device", "device"),
        ("logdir", "log_root"),
        ("ckpt", "ckpt_root"),
        ("metrics", "reports_root"),
        ("reports_dir", "reports_root"),
        ("model_name", "model_name"),
    ]:
        val = getattr(args, arg_name)
        defval = getattr(defaults, arg_name)
        if val is not None and val != defval:
            if arg_name == "ckpt":
                cfg[cfg_key] = str(Path(val).parent)
            elif arg_name in ("metrics", "reports_dir"):
                cfg[cfg_key] = str(Path(val).parent)
            else:
                cfg[cfg_key] = val
    cfg["_config_path"] = args.config
    return cfg


def is_improved(best_value, current, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < (best_value - min_delta)
    return current > (best_value + min_delta)


def save_checkpoint(payload: dict, path: Path):
    torch.save(payload, str(path))


# ----------------- model -----------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # logits


def build_model(model_name: str, num_classes: int, img_size: int):
    """
    Returns (model, default_target_layer)
    """
    if model_name == "smallcnn":
        m = SmallCNN(num_classes=num_classes)
        return m, "conv2"

    if model_name == "resnet18_cifar":
        # Start from vanilla resnet18 but adapt for CIFAR (32x32)
        m = models.resnet18(weights=None)
        # 3x3 conv, stride=1, padding=1 instead of 7x7/stride=2, and remove maxpool
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        # replace classifier
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, "layer4"

    if model_name == "resnet18_imagenet":
        # Use ImageNet weights and resize input to 224
        try:
            w = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            w = None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, "layer4"

    raise ValueError(f"Unknown model_name: {model_name}")


# ----------------- data -----------------
def get_transforms_for(dataset_name: str, img_size: int, mean, std, train: bool):
    tfms = []
    if dataset_name in {"cifar10"}:
        if train:
            # light augments for CIFAR
            if img_size == 32:
                tfms += [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                tfms += [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                ]
        else:
            tfms += [transforms.Resize((img_size, img_size))]
        tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tfms)
    
    if dataset_name == "eurosat":
        # Work at 224px (standard backbones) + light aug for train
        if train:
            tfms += [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
            ]
        else:
            tfms += [transforms.Resize((img_size, img_size))]
        tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tfms)

    # fashion-mnist / mnist (grayscale)
    # fashion-mnist / mnist (grayscale)
    m, s = float(mean[0]), float(std[0])
    tfms = [transforms.ToTensor(), transforms.Normalize((m,), (s,))]
    return transforms.Compose(tfms)


def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    img_size: int,
    mean,
    std,
):
    root = Path(data_dir)
    g = torch.Generator().manual_seed(seed)

    if dataset_name == "fashion-mnist":
        train_tf = get_transforms_for("fashion-mnist", img_size, mean, std, train=True)
        eval_tf  = get_transforms_for("fashion-mnist", img_size, mean, std, train=False)
        train_ds = tv.datasets.FashionMNIST(root=root, train=True,  download=True, transform=train_tf)
        test_ds  = tv.datasets.FashionMNIST(root=root, train=False, download=True, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, generator=g)
        val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = train_ds.classes
        return train_loader, val_loader, test_loader, classes

    elif dataset_name == "mnist":
        train_tf = get_transforms_for("mnist", img_size, mean, std, train=True)
        eval_tf  = get_transforms_for("mnist", img_size, mean, std, train=False)
        train_ds = tv.datasets.MNIST(root=root, train=True,  download=True, transform=train_tf)
        test_ds  = tv.datasets.MNIST(root=root, train=False, download=True, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, generator=g)
        val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = train_ds.classes
        return train_loader, val_loader, test_loader, classes

    elif dataset_name == "cifar10":
        train_tf = get_transforms_for("cifar10", img_size, mean, std, train=True)
        eval_tf  = get_transforms_for("cifar10", img_size, mean, std, train=False)
        train_ds = tv.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
        test_ds  = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, generator=g)
        val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = train_ds.classes
        return train_loader, val_loader, test_loader, classes

    elif dataset_name == "eurosat":
        # Expect prepared split at: data/eurosat_custom/{train,val,test}/{class}/*.jpg
        train_tf = get_transforms_for("eurosat", img_size, mean, std, train=True)
        eval_tf  = get_transforms_for("eurosat", img_size, mean, std, train=False)

        train_ds = ImageFolder(root / "eurosat_custom" / "train", transform=train_tf)
        val_ds   = ImageFolder(root / "eurosat_custom" / "val",   transform=eval_tf)
        test_ds  = ImageFolder(root / "eurosat_custom" / "test",  transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, generator=g)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = train_ds.classes
        return train_loader, val_loader, test_loader, classes

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ----------------- train/eval -----------------
def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        b = yb.size(0)
        loss_sum += loss.item() * b
        acc_sum += accuracy(logits, yb) * b
        n += b
    return loss_sum / n, acc_sum / n


@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        b = yb.size(0)
        loss_sum += loss.item() * b
        acc_sum += accuracy(logits, yb) * b
        n += b
    return loss_sum / n, acc_sum / n


@torch.no_grad()
def confusion_matrix_report(
    model,
    test_loader,
    device,
    classes,
    reports_dir: Path,
    metrics_path: Path,
    title_prefix: str,
):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(yb)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    num_classes = len(classes)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes)
    cm = cm_metric(all_preds, all_targets).numpy()
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{title_prefix} Confusion Matrix (row-normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center", fontsize=8
            )
    fig.tight_layout()
    fig_path = reports_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print("Saved figure to:", fig_path)

    np.save(reports_dir / "confusion_matrix_counts.npy", cm)
    np.save(reports_dir / "confusion_matrix_norm.npy", cm_norm)

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    metrics.update(
        {
            "confusion_matrix_counts_path": str(
                reports_dir / "confusion_matrix_counts.npy"
            ),
            "confusion_matrix_norm_path": str(
                reports_dir / "confusion_matrix_norm.npy"
            ),
            "confusion_matrix_figure": str(fig_path),
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


# ----------------- main -----------------
def main():
    parser = build_argparser()
    args = parser.parse_args()
    seed_all(args.seed)

    base_cfg = load_yaml(args.config)
    cfg = merge_cli_over_config_with_defaults(base_cfg, args, parser)

    dataset = cfg["dataset"]
    model_name = cfg.get("model_name", "smallcnn")

    img_size = int(
        cfg.get("img_size", 28 if dataset in ["fashion-mnist", "mnist"] else 32)
    )
    mean = cfg.get("mean", None)
    std = cfg.get("std", None)

    # defaults for grayscale datasets
    if dataset in ["fashion-mnist", "mnist"]:
        if mean is None or std is None:
            if dataset == "fashion-mnist":
                mean, std = [0.2860], [0.3530]
            else:
                mean, std = [0.1307], [0.3081]
    # defaults for cifar10
    if dataset == "cifar10" and (mean is None or std is None):
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

    # defaults for eurosat (we use ImageNet stats at 224px)
    if dataset == "eurosat":
        img_size = int(cfg.get("img_size", 224))
        if mean is None or std is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    device = get_device(cfg["device"])
    print("device:", device)

    run_id = f'{cfg["dataset"]}_{int(time.time())}'
    LOG_DIR = Path(cfg["log_root"]) / run_id
    CKPTS_DIR = Path(cfg["ckpt_root"]) / run_id
    REPORTS_DIR = Path(cfg["reports_root"]) / run_id
    for d in (LOG_DIR, CKPTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    effective_cfg = deepcopy(cfg)
    effective_cfg["run_id"] = run_id
    with open(REPORTS_DIR / "config_effective.yaml", "w") as f:
        yaml.safe_dump(effective_cfg, f)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset,
        cfg["data_dir"],
        cfg["batch_size"],
        cfg["num_workers"],
        cfg["seed"],
        img_size,
        mean,
        std,
        )

    if dataset == "eurosat":
        # ImageFolder exposes .samples -> list of (path, label)
        targets = np.array([y for _, y in train_loader.dataset.samples])
        class_counts = np.bincount(targets, minlength=len(classes))
        class_weights = (class_counts.sum() / (len(classes) * class_counts))
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    model, default_target_layer = build_model(
        model_name, num_classes=len(classes), img_size=img_size
    )
    model = model.to(device)

    opt_name = str(cfg.get("optimizer", "adam")).lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=cfg["weight_decay"],
            nesterov=True,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    writer = SummaryWriter(log_dir=str(LOG_DIR))

    monitor = cfg["early_stop"]["monitor"]
    mode = cfg["early_stop"]["mode"]
    patience = int(cfg["early_stop"]["patience"])
    min_delta = float(cfg["early_stop"]["min_delta"])

    best_val = float("inf") if mode == "min" else -float("inf")
    epochs_no_improve = 0

    ckpt_last = CKPTS_DIR / "last.ckpt"
    ckpt_best = CKPTS_DIR / "best.ckpt"

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, device, optimizer, loss_fn
        )
        va_loss, va_acc = eval_one_epoch(model, val_loader, device, loss_fn)
        scheduler.step(va_loss)

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f}"
            + f" | val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        mon_value = va_loss if monitor == "val_loss" else va_acc

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": va_acc,
            "val_loss": va_loss,
            "dataset": cfg["dataset"],
            "classes": classes,
            "config_path": cfg.get("_config_path"),
            "meta": {
                "dataset": dataset,
                "model_name": model_name,
                "img_size": img_size,
                "mean": mean,
                "std": std,
                "default_target_layer": default_target_layer,
            },
        }
        save_checkpoint(payload, ckpt_last)

        if is_improved(best_val, mon_value, mode, min_delta):
            best_val = mon_value
            epochs_no_improve = 0
            save_checkpoint(payload, ckpt_best)
            best_json = {
                "epoch": epoch,
                "monitor": monitor,
                "mode": mode,
                "best_value": float(best_val),
                "val_acc": float(va_acc),
                "val_loss": float(va_loss),
                "ckpt_path": str(ckpt_best),
                "meta": {
                    "dataset": dataset,
                    "model_name": model_name,
                    "img_size": img_size,
                    "mean": mean,
                    "std": std,
                    "default_target_layer": default_target_layer,
                },
            }
            with open(REPORTS_DIR / "best.json", "w") as f:
                json.dump(best_json, f, indent=2)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    writer.close()
    print(f"Best {monitor}: {best_val:.4f}")

    # Use best checkpoint for reports
    best_ckpt = torch.load(str(ckpt_best), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    metrics_path = REPORTS_DIR / "metrics.json"
    confusion_matrix_report(
        model,
        test_loader,
        device,
        classes,
        reports_dir=REPORTS_DIR / "figures",
        metrics_path=metrics_path,
        title_prefix=cfg["dataset"].replace("-", " ").title(),
    )

    metrics = {
        "dataset": cfg["dataset"],
        "epochs_ran": epoch,
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "best_monitor": monitor,
        "best_mode": mode,
        "best_value": float(best_val),
        "logs_dir": str(LOG_DIR),
        "ckpts_dir": str(CKPTS_DIR),
        "reports_dir": str(REPORTS_DIR),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to:", metrics_path)
    print("Best checkpoint:", ckpt_best)


if __name__ == "__main__":
    main()
