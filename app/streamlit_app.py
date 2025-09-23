import datetime as dt
import random
from pathlib import Path
import os
import hashlib
import requests

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torchcam.methods import GradCAM, GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


# ---------- Small utilities ----------
def get_device(choice="auto"):
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_latest_best_ckpt():
    ckpts = sorted(
        Path("checkpoints").rglob("best.ckpt"), key=lambda p: p.stat().st_mtime
    )
    return ckpts[-1] if ckpts else None


def denorm_to_pil(x, mean, std):
    """
    x: torch.Tensor CxHxW (normalized), mean/std lists
    returns PIL.Image (RGB)
    """
    x = x.detach().cpu().clone()
    if len(mean) == 1:
        # grayscale
        m, s = float(mean[0]), float(std[0])
        x = x * s + m  # de-normalize
        x = x.clamp(0, 1)
        # convert to RGB for overlay convenience
        pil = T.ToPILImage()(x)
        pil = pil.convert("RGB")
        return pil
    else:
        mean = torch.tensor(mean)[:, None, None]
        std = torch.tensor(std)[:, None, None]
        x = x * std + mean
        x = x.clamp(0, 1)
        return T.ToPILImage()(x)


DATASET_CLASSES = {
    "fashion-mnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}


@st.cache_resource
def load_raw_dataset(name: str, root="data"):
    """Load the test split with ToTensor() only (for preview)."""
    tt = T.ToTensor()
    if name == "fashion-mnist":
        ds = FashionMNIST(root=root, train=False, download=True, transform=tt)
    elif name == "mnist":
        ds = MNIST(root=root, train=False, download=True, transform=tt)
    elif name == "cifar10":
        ds = CIFAR10(root=root, train=False, download=True, transform=tt)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    classes = getattr(ds, "classes", None) or [str(i) for i in range(10)]
    return ds, classes


def pil_from_tensor(img_tensor, grayscale_to_rgb=True):
    pil = T.ToPILImage()(img_tensor)
    if grayscale_to_rgb and img_tensor.ndim == 3 and img_tensor.shape[0] == 1:
        pil = pil.convert("RGB")
    return pil


@st.cache_data(ttl=5, show_spinner=False)
def list_ckpts(root_dir: str, recursive: bool = True, filter: str = ""):
    """Return (labels, paths) sorted by mtime desc."""
    root = Path(root_dir)
    if not root.exists():
        return [], []
    files = sorted(
        (root.rglob("*.ckpt") if recursive else root.glob("*.ckpt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    files = [p for p in files if filter in str(p)]
    labels = []
    for p in files:
        rel = p.relative_to(root)
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        labels.append(f"{rel}  •  {mtime}")
    return labels, [str(p) for p in files]


# ---------- Your SmallCNN (for FMNIST) ----------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
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
        return self.fc(x)


# ---------- Load model + meta from checkpoint ----------
def load_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    classes = ckpt.get("classes", None)
    meta = ckpt.get("meta", {})
    num_classes = len(classes) if classes else 10
    model_name = meta.get("model_name", "smallcnn")

    if model_name == "smallcnn":
        model = SmallCNN(num_classes=num_classes).to(device)
        default_target_layer = "conv2"
    elif model_name == "resnet18_cifar":
        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m.to(device)
        default_target_layer = "layer4"
    elif model_name == "resnet18_imagenet":
        try:
            w = tvm.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            w = None
        m = tvm.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m.to(device)
        default_target_layer = "layer4"
    else:
        raise ValueError(f"Unknown model_name in ckpt: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # ensure meta has defaults
    meta.setdefault("default_target_layer", default_target_layer)
    return model, classes, meta


def build_transform_from_meta(meta):
    img_size = int(meta.get("img_size", 28))
    mean = meta.get("mean", [0.2860])  # FMNIST fallback
    std = meta.get("std", [0.3530])
    if len(mean) == 1:
        return T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )


def predict_and_cam(model, x, device, target_layer, topk=3, method="Grad-CAM"):
    """
    x: Tensor [1,C,H,W] normalized
    returns: list of dicts: {rank, class_index, prob, cam_tensor(H,W)}
    """
    cam_cls = GradCAM if method == "Grad-CAM" else GradCAMpp
    cam_extractor = cam_cls(model, target_layer=target_layer)

    logits = model(x.to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    top_vals, top_idxs = probs.topk(topk)

    results = []
    for rank, (p, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist())):
        retain = rank < topk - 1
        cams = cam_extractor(idx, logits, retain_graph=retain)  # list
        cam = cams[0].detach().cpu()  # [H,W] at feature-map resolution
        results.append(
            {"rank": rank + 1, "class_index": int(idx), "prob": float(p), "cam": cam}
        )
    return results, probs


def overlay_pil(base_pil_rgb: Image.Image, cam_tensor, alpha=0.5):
    # cam_tensor: torch.Tensor HxW in [0,1] (we'll min-max it)
    cam = cam_tensor.clone()
    cam -= cam.min()
    cam = cam / (cam.max() + 1e-8)
    heat = T.ToPILImage()(cam)  # single-channel PIL
    return overlay_mask(base_pil_rgb, heat, alpha=alpha)


# ---------- UI ----------
st.set_page_config(page_title="Grad-CAM Demo", page_icon="🔍", layout="wide")
st.title("🔍 Grad-CAM Demo — upload an image, get top-k + heatmaps")

# Sidebar: checkpoint + options
with st.sidebar:
    st.header("Settings")

    source = st.radio("Checkpoint source", ["Local folder", "Remote URL/Presets"], index=0)
    show_best_only = st.checkbox("Only show best.ckpt", value=False)
    filter = "best" if show_best_only else ""

    ckpt_path = None

    if source == "Local folder":
        ckpt_root = st.text_input("Checkpoint folder", value="saved_checkpoints")
        if st.button("Refresh list", use_container_width=True):
            st.cache_data.clear()
        labels, paths = list_ckpts(ckpt_root, recursive=True, filter=filter)
        if not paths:
            st.warning(f"No matching checkpoints found under: {ckpt_root}")
        else:
            sel_label = st.selectbox("Select a checkpoint", options=labels, index=0)
            ckpt_path = paths[labels.index(sel_label)]
    else:
        st.subheader("Remote checkpoints (GitHub Releases)")
        dest_dir = st.text_input("Download to folder", value="saved_checkpoints")

        def download_release_asset(url: str, dest_dir: str = "saved_checkpoints") -> str:
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            fname = Path(url).name or f"asset_{url_hash}.ckpt"
            if not fname.endswith(".ckpt"):
                fname = f"{fname}.ckpt"
            local_path = Path(dest_dir) / f"{url_hash}_{fname}"
            if local_path.exists() and local_path.stat().st_size > 0:
                return str(local_path)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return str(local_path)

        presets = st.secrets.get("release_checkpoints", {}) if hasattr(st, "secrets") else {}
        preset_names = list(presets.keys())
        preset_sel = st.selectbox("Preset release asset", options=["(none)"] + preset_names, index=0) if preset_names else "(none)"
        url_input = st.text_input("Or paste asset URL", value="")
        if st.button("Download checkpoint", use_container_width=True):
            url = presets.get(preset_sel, "") if preset_sel != "(none)" else url_input.strip()
            if not url:
                st.warning("Provide a preset or paste a URL")
            else:
                try:
                    path_dl = download_release_asset(url, dest_dir=dest_dir)
                    st.success(f"Downloaded to: {path_dl}")
                    ckpt_path = path_dl
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Download failed: {e}")

    st.caption(f"Selected: {ckpt_path}")

    with st.expander("Checkpoint meta preview", expanded=False):
        try:
            if ckpt_path:
                m, c, meta_preview = load_model_from_ckpt(Path(ckpt_path), device="cpu")
                st.json(
                    {
                        "dataset": meta_preview.get("dataset"),
                        "model_name": meta_preview.get("model_name"),
                        "img_size": meta_preview.get("img_size"),
                        "target_layer": meta_preview.get("default_target_layer"),
                    }
                )
            else:
                st.info("No checkpoint selected yet.")
        except Exception as e:
            st.info(f"Could not read meta: {e}")

    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    method = st.selectbox("CAM method", ["Grad-CAM", "Grad-CAM++"], index=0)
    topk = st.slider("Top-k classes", min_value=1, max_value=10, value=3, step=1)
    alpha = st.slider(
        "Overlay alpha", min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )

# Load model/meta
if not ckpt_path or not Path(ckpt_path).exists():
    st.info("Select a local checkpoint or download one from GitHub Releases above.")
    st.stop()

device = get_device(device_choice)
model, classes, meta = load_model_from_ckpt(Path(ckpt_path), device)
tf = build_transform_from_meta(meta)
target_layer = meta.get("default_target_layer", "conv2")

# Main: uploader
# Main: uploader OR dataset sample
st.subheader("1) Provide an image")
uploaded = st.file_uploader(
    "Upload PNG/JPG (or pick a sample below)", type=["png", "jpg", "jpeg"]
)

with st.expander("…or pick a sample from this model's dataset", expanded=False):
    # Lock to the dataset the model was trained on
    ds_default = meta.get("dataset", "fashion-mnist")
    ds, ds_classes = load_raw_dataset(ds_default, root="data")

    # Targets for filtering
    targets = np.array(getattr(ds, "targets", [ds[i][1] for i in range(len(ds))]))

    # Class filter (optional)
    class_opts = ["(any)"] + list(ds_classes)
    class_sel = st.selectbox("Class filter", options=class_opts, index=0)

    if class_sel == "(any)":
        filtered_idx = np.arange(len(ds))
    else:
        class_id = ds_classes.index(class_sel)
        filtered_idx = np.nonzero(targets == class_id)[0]

    # Index picker within the filtered subset
    if len(filtered_idx) == 0:
        st.info("No samples found for this class.")
        sample_img = None
    else:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            idx_pos = st.slider(
                "Pick index (within filtered samples)",
                0,
                max(0, len(filtered_idx) - 1),
                0,
            )
        with col_r:
            if st.button("Pick random"):
                idx_pos = random.randrange(len(filtered_idx))

        raw_idx = int(filtered_idx[idx_pos])
        img_tensor, label = ds[raw_idx]
        sample_img = pil_from_tensor(img_tensor, grayscale_to_rgb=True)

        # SMALL preview (set width)
        caption = (
            f"Sample • {ds_default} • " f"class={ds_classes[label]} • " f"idx={raw_idx}"
        )
        st.image(
            sample_img,
            caption=caption,
            width=160,
            use_container_width=False,
        )

# Decide the input image used downstream
if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
elif "sample_img" in locals() and sample_img is not None:
    pil = sample_img
else:
    st.info("Upload an image or open the sample picker above.")
    st.stop()

col_in, col_cfg = st.columns([2, 1])

with col_in:
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
    elif sample_img is not None:
        pil = sample_img
    else:
        st.info("Upload an image or check 'Use a sample image'.")
        st.stop()

    st.image(pil, caption="Input", use_container_width=True)

with col_cfg:
    st.markdown("**Model meta**")
    st.json(
        {
            "dataset": meta.get("dataset"),
            "model_name": meta.get("model_name"),
            "img_size": meta.get("img_size"),
            "target_layer": target_layer,
            "mean": meta.get("mean"),
            "std": meta.get("std"),
            "classes": (
                classes
                if classes and len(classes) <= 10
                else f"{len(classes) if classes else 'N/A'} classes"
            ),
        }
    )

# Prepare tensor + denormalized PIL base for overlay
x = tf(pil)  # CxHxW normalized
x_batched = x.unsqueeze(0)  # 1xCxHxW
base_pil = denorm_to_pil(x, meta.get("mean", [0.2860]), meta.get("std", [0.3530]))

# Predict + CAM
with st.spinner("Running inference + Grad-CAM..."):
    try:
        cam_results, probs = predict_and_cam(
            model, x_batched, device, target_layer, topk=topk, method=method
        )
    except Exception as e:
        st.error(
            f"Grad-CAM failed. Target layer likely incorrect."
            f"\nLayer: {target_layer}\nError: {e}"
        )
        st.stop()

# Top-k table
st.subheader("2) Top-k predictions")
rows = []
for r in cam_results:
    name = classes[r["class_index"]] if classes else str(r["class_index"])
    rows.append(
        {
            "rank": r["rank"],
            "class": name,
            "index": r["class_index"],
            "prob": round(r["prob"], 4),
        }
    )
st.dataframe(rows, use_container_width=True)

# Overlays
st.subheader("3) Grad-CAM overlays")
cols = st.columns(len(cam_results))
for c, r in zip(cols, cam_results):
    name = classes[r["class_index"]] if classes else str(r["class_index"])
    ov = overlay_pil(base_pil, r["cam"], alpha=alpha)
    with c:
        st.image(
            ov,
            caption=f"Top{r['rank']}: {name} ({r['prob']:.3f})",
            use_container_width=True,
        )
