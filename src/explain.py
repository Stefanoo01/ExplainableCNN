import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision import models as tvm
from torchvision import transforms

from src.train import SmallCNN, get_device


def build_argparser():
    p = argparse.ArgumentParser(description="Grad-CAM explanations")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.ckpt")
    p.add_argument("--image", type=str, required=True, help="Path to an input image")
    p.add_argument(
        "--dataset",
        choices=["fashion-mnist", "mnist", "cifar10"],
        default="fashion-mnist",
        help="Used to apply the right normalization and class names",
    )
    p.add_argument(
        "--target-layer",
        type=str,
        default="conv2",
        help="Layer to attach CAMs (e.g., 'conv2' for SmallCNN, 'layer4' for ResNet)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to store results; defaults near the checkpoint",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--topk", type=int, default=3, help="How many top classes to render")
    return p


def get_transforms_from_meta(meta):
    img_size = int(meta.get("img_size", 28))
    mean = meta.get("mean", [0.2860])  # fallback FMNIST
    std = meta.get("std", [0.3530])

    # channels: grayscale if mean/std length==1, else RGB
    if len(mean) == 1:
        tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return tf


def denorm_to_pil(x: torch.Tensor, mean, std) -> Image.Image:
    """
    x: normalized tensor CxHxW
    mean/std: list(s) from meta
    returns: PIL RGB image for overlay
    """
    x = x.detach().cpu().clone()
    if len(mean) == 1:  # grayscale
        m, s = float(mean[0]), float(std[0])
        x = x * s + m
        x = x.clamp(0, 1)
        pil = transforms.ToPILImage()(x)  # grayscale PIL
        return pil.convert("RGB")
    else:  # RGB
        mean_t = torch.tensor(mean)[:, None, None]
        std_t = torch.tensor(std)[:, None, None]
        x = x * std_t + mean_t
        x = x.clamp(0, 1)
        return transforms.ToPILImage()(x)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get("classes", None)
    meta = ckpt.get("meta", {})
    num_classes = len(classes) if classes else 10
    model_name = meta.get("model_name", "smallcnn")

    if model_name == "smallcnn":
        model = SmallCNN(num_classes=num_classes).to(device)
    elif model_name == "resnet18_cifar":
        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m.to(device)
    elif model_name == "resnet18_imagenet":
        try:
            w = tvm.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            w = None
        m = tvm.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m.to(device)
    else:
        raise ValueError(f"Unknown model in ckpt: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes, meta


def run_gradcam(
    model,
    target_layer,
    img_tensor,
    device,
    classes,
    outdir: Path,
    topk=3,
    base_pil_rgb: Image.Image = None,
):
    """
    img_tensor: CxHxW normalized (not batched)
    base_pil_rgb: PIL image already denormalized & RGB for overlay (optional).
                  If None, will min-max scale from img_tensor (last-resort).
    """
    model.eval()
    x = img_tensor.to(device).unsqueeze(0)  # [1,C,H,W]
    H, W = img_tensor.shape[-2:]
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # forward once to get top-k
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    top_vals, top_idxs = probs.topk(topk)

    if base_pil_rgb is None:
        # Fallback: simple min-max scaling (works but less faithful than denorm)
        xx = img_tensor.detach().cpu()
        xx = (xx - xx.min()) / (xx.max() - xx.min() + 1e-8)
        base_pil_rgb = transforms.ToPILImage()(xx)
        if xx.shape[0] == 1:
            base_pil_rgb = base_pil_rgb.convert("RGB")

    results = []
    for rank, (score, cls_idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist())):
        retain = rank < topk - 1
        cams = cam_extractor(int(cls_idx), logits, retain_graph=retain)
        cam = cams[0].detach().cpu()  # [h,w]
        cam_up = TF.resize(cam.unsqueeze(0), size=[H, W])[0]  # upsample to input size

        heat = transforms.ToPILImage()(cam_up)
        overlay = overlay_mask(base_pil_rgb, heat, alpha=0.6)

        out_png = (
            outdir / f"gradcam_top{rank+1}_class{cls_idx}_"
            + f"{classes[cls_idx] if classes else cls_idx}.png"
        )
        overlay.save(out_png)

        results.append(
            {
                "rank": rank + 1,
                "class_index": int(cls_idx),
                "class_name": classes[cls_idx] if classes else str(cls_idx),
                "prob": float(score),
                "file": str(out_png),
            }
        )

    with open(outdir / "summary.json", "w") as f:
        json.dump({"topk": results}, f, indent=2)

    print("Saved:", outdir)
    return results


def main():
    args = build_argparser().parse_args()
    device = get_device(args.device)

    ckpt_path = Path(args.ckpt)

    # outdir default
    if args.outdir is None:
        run_id = ckpt_path.parent.name
        outdir = ckpt_path.parent.parent.parent / "reports" / run_id / "explain"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load model+meta first
    model, classes, meta = load_model(str(ckpt_path), device)

    # 2) build tf from meta
    tf = get_transforms_from_meta(meta)

    # 3) load and transform image
    pil = Image.open(args.image).convert("RGB")
    x = tf(pil)  # CxHxW normalized

    # 4) make a denormalized RGB base image for overlay
    base_pil = denorm_to_pil(x, meta.get("mean", [0.2860]), meta.get("std", [0.3530]))

    # 5) target layer (CLI overrides meta default)
    target_layer = args.target_layer or meta.get("default_target_layer", "conv2")

    # 6) run Grad-CAM
    results = run_gradcam(
        model,
        target_layer,
        x,
        device,
        classes,
        outdir,
        topk=args.topk,
        base_pil_rgb=base_pil,
    )

    # 7) print summary
    for r in results:
        print(f"Top{r['rank']}: {r['class_name']} ({r['prob']:.3f}) -> {r['file']}")


if __name__ == "__main__":
    main()
