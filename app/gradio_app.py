import datetime as dt
import random
from pathlib import Path
import os
import hashlib
import requests
import json
import tempfile

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torchcam.methods import GradCAM, GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

# Global state for model and configuration
app_state = {
    "model": None,
    "classes": None,
    "meta": None,
    "transform": None,
    "target_layer": None,
    "dataset": None,
    "dataset_classes": None
}

custom_theme = gr.themes.Soft(
    primary_hue="green",     # main brand color
    secondary_hue="green",  # accent color
    neutral_hue="slate"       # backgrounds/borders/text neutrals
)

def download_release_asset(url: str, dest_dir: str = "saved_checkpoints") -> str:
    """Download a remote checkpoint to dest_dir and return its local path."""
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


def load_release_presets() -> dict:
    """Load release preset URLs from multiple sources."""
    # Try environment variable containing JSON mapping
    env_json = os.environ.get("RELEASE_CKPTS_JSON", "").strip()
    if env_json:
        try:
            data = json.loads(env_json)
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            pass
    
    # Try local JSON files for dev
    for rel in (".streamlit/presets.json", "presets.json"):
        p = Path(rel)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    if "release_checkpoints" in data and isinstance(data["release_checkpoints"], dict):
                        return dict(data["release_checkpoints"])
                    return dict(data)
            except Exception:
                pass
    
    return {}


def get_device(choice="auto"):
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def denorm_to_pil(x, mean, std):
    """Convert normalized tensor to PIL Image."""
    x = x.detach().cpu().clone()
    if len(mean) == 1:
        # grayscale
        m, s = float(mean[0]), float(std[0])
        x = x * s + m
        x = x.clamp(0, 1)
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
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}


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
    meta.setdefault("default_target_layer", default_target_layer)
    return model, classes, meta


def build_transform_from_meta(meta):
    img_size = int(meta.get("img_size", 28))
    mean = meta.get("mean", [0.2860])
    std = meta.get("std", [0.3530])
    if len(mean) == 1:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def predict_and_cam(model, x, device, target_layer, topk=3, method="Grad-CAM"):
    """Predict and generate CAM for top-k classes."""
    cam_cls = GradCAM if method == "Grad-CAM" else GradCAMpp
    cam_extractor = cam_cls(model, target_layer=target_layer)

    logits = model(x.to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    top_vals, top_idxs = probs.topk(topk)

    results = []
    for rank, (p, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist())):
        retain = rank < topk - 1
        cams = cam_extractor(idx, logits, retain_graph=retain)
        cam = cams[0].detach().cpu()
        results.append({
            "rank": rank + 1,
            "class_index": int(idx),
            "prob": float(p),
            "cam": cam
        })
    return results, probs


def overlay_pil(base_pil_rgb: Image.Image, cam_tensor, alpha=0.5):
    """Create overlay of CAM on base image."""
    cam = cam_tensor.clone()
    cam -= cam.min()
    cam = cam / (cam.max() + 1e-8)
    heat = T.ToPILImage()(cam)
    return overlay_mask(base_pil_rgb, heat, alpha=alpha)


# Gradio interface functions
def load_checkpoint_from_url(url, preset_name):
    """Load checkpoint from URL or preset."""
    presets = load_release_presets()
    
    if preset_name and preset_name != "None":
        url = presets.get(preset_name, "")
    
    if not url:
        return "❌ No URL provided", "", ""
    
    try:
        ckpt_path = download_release_asset(url)
        device = get_device("cpu")
        model, classes, meta = load_model_from_ckpt(Path(ckpt_path), device)
        
        # Update global state
        app_state["model"] = model
        app_state["classes"] = classes
        app_state["meta"] = meta
        app_state["transform"] = build_transform_from_meta(meta)
        app_state["target_layer"] = meta.get("default_target_layer", "conv2")
        
        # Load dataset for samples
        ds_name = meta.get("dataset", "fashion-mnist")
        try:
            dataset, dataset_classes = load_raw_dataset(ds_name)
            app_state["dataset"] = dataset
            app_state["dataset_classes"] = dataset_classes
        except:
            app_state["dataset"] = None
            app_state["dataset_classes"] = None
        
        meta_info = {
            "dataset": meta.get("dataset"),
            "model_name": meta.get("model_name"),
            "img_size": meta.get("img_size"),
            "target_layer": app_state["target_layer"],
            "mean": meta.get("mean"),
            "std": meta.get("std"),
            "classes": len(classes) if classes else "N/A"
        }
        
        # Create class choices for filter
        class_choices = ["(any)"] + (dataset_classes if app_state["dataset"] else [])
        max_samples = len(dataset) - 1 if app_state["dataset"] else 0
        
        return (f"✅ Loaded: {ckpt_path}", json.dumps(meta_info, indent=2), 
                gr.update(visible=True), gr.update(choices=class_choices, value="(any)", visible=True),
                gr.update(visible=True, maximum=max_samples, value=0), gr.update(visible=True, value=""))
    
    except Exception as e:
        return f"❌ Failed: {str(e)}", "", gr.update(visible=False), gr.update(choices=["(any)"], value="(any)"), gr.update(visible=False), gr.update(choices=["(any)"], value="(any)"), gr.update(visible=False)


def load_checkpoint_from_file(file):
    """Load checkpoint from uploaded file."""
    if file is None:
        return "❌ No file uploaded", "", ""
    
    try:
        # Save uploaded file temporarily
        Path("saved_checkpoints").mkdir(parents=True, exist_ok=True)
        with open(file.name, "rb") as f:
            content = f.read()
        
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        base_name = Path(file.name).name
        if not base_name.endswith(".ckpt"):
            base_name = f"{base_name}.ckpt"
        local_path = Path("saved_checkpoints") / f"{content_hash}_{base_name}"
        
        with open(local_path, "wb") as f:
            f.write(content)
        
        device = get_device("cpu")
        model, classes, meta = load_model_from_ckpt(local_path, device)
        
        # Update global state
        app_state["model"] = model
        app_state["classes"] = classes
        app_state["meta"] = meta
        app_state["transform"] = build_transform_from_meta(meta)
        app_state["target_layer"] = meta.get("default_target_layer", "conv2")
        
        # Load dataset for samples
        ds_name = meta.get("dataset", "fashion-mnist")
        try:
            dataset, dataset_classes = load_raw_dataset(ds_name)
            app_state["dataset"] = dataset
            app_state["dataset_classes"] = dataset_classes
        except:
            app_state["dataset"] = None
            app_state["dataset_classes"] = None
        
        meta_info = {
            "dataset": meta.get("dataset"),
            "model_name": meta.get("model_name"),
            "img_size": meta.get("img_size"),
            "target_layer": app_state["target_layer"],
            "mean": meta.get("mean"),
            "std": meta.get("std"),
            "classes": len(classes) if classes else "N/A"
        }
        
        # Create class choices for filter
        class_choices = ["(any)"] + (dataset_classes if app_state["dataset"] else [])
        max_samples = len(dataset) - 1 if app_state["dataset"] else 0
        
        return (f"✅ Loaded: {local_path}", json.dumps(meta_info, indent=2), 
                gr.update(visible=True), gr.update(choices=class_choices, value="(any)", visible=True),
                gr.update(visible=True, maximum=max_samples, value=0), gr.update(visible=True, value=""))
    
    except Exception as e:
        return f"❌ Failed: {str(e)}", "", gr.update(visible=False)


def get_random_sample(class_filter="(any)"):
    """Get a random sample from the (optionally filtered) dataset."""
    if app_state["dataset"] is None:
        return None, "No dataset loaded", gr.update(visible=False)

    dataset = app_state["dataset"]
    dataset_classes = app_state["dataset_classes"]

    # Build candidate indices according to filter
    if class_filter != "(any)":
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        class_id = dataset_classes.index(class_filter)
        filtered_indices = np.where(targets == class_id)[0]
        if len(filtered_indices) == 0:
            return None, f"No samples found for class: {class_filter}", gr.update(visible=True, maximum=0, value=0)
        actual_idx = int(random.choice(filtered_indices))
        # slider index is relative to the filtered list length
        slider_max = len(filtered_indices) - 1
        slider_value = int(np.where(filtered_indices == actual_idx)[0][0])
    else:
        actual_idx = random.randint(0, len(dataset) - 1)
        slider_max = len(dataset) - 1
        slider_value = actual_idx

    img_tensor, label = dataset[actual_idx]
    sample_img = pil_from_tensor(img_tensor, grayscale_to_rgb=True)
    sample_img = double_height(sample_img) 
    class_name = dataset_classes[label] if dataset_classes else str(label)
    caption = f"Sample {actual_idx} from {app_state['meta'].get('dataset', 'dataset')} • class: {class_name}"

    # Update slider to the picked index inside the current filter's range
    return sample_img, caption, gr.update(visible=True, maximum=slider_max, value=slider_value)


def get_sample_by_index(idx, class_filter):
    """Get a specific sample by index with optional class filtering."""
    if app_state["dataset"] is None:
        return None, "No dataset loaded"
    
    dataset = app_state["dataset"]
    dataset_classes = app_state["dataset_classes"]
    
    # Apply class filter
    if class_filter != "(any)":
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        class_id = dataset_classes.index(class_filter)
        filtered_indices = np.where(targets == class_id)[0]
        
        if len(filtered_indices) == 0:
            return None, f"No samples found for class: {class_filter}"
        
        # Clamp index to filtered range
        idx = max(0, min(idx, len(filtered_indices) - 1))
        actual_idx = filtered_indices[idx]
    else:
        # Clamp index to dataset range
        idx = max(0, min(idx, len(dataset) - 1))
        actual_idx = idx
    
    img_tensor, label = dataset[actual_idx]
    sample_img = pil_from_tensor(img_tensor, grayscale_to_rgb=True)
    sample_img = double_height(sample_img)
    class_name = dataset_classes[label] if dataset_classes else str(label)
    caption = f"Sample {actual_idx} from {app_state['meta'].get('dataset', 'dataset')} • class: {class_name}"
    
    return sample_img, caption


def update_class_filter(class_filter):
    """Update the slider range when class filter changes."""
    if app_state["dataset"] is None:
        return gr.update(visible=False, maximum=0, value=0)
    
    dataset = app_state["dataset"]
    dataset_classes = app_state["dataset_classes"]
    
    if class_filter == "(any)":
        max_idx = len(dataset) - 1
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        class_id = dataset_classes.index(class_filter)
        filtered_indices = np.where(targets == class_id)[0]
        max_idx = len(filtered_indices) - 1 if len(filtered_indices) > 0 else 0
    
    return gr.update(visible=True, maximum=max_idx, value=0)


def double_height(img: Image.Image) -> Image.Image:
    """Return a copy of the image with doubled height."""
    w, h = img.size
    return img.resize((w * 10, h * 10), Image.Resampling.NEAREST)


def process_image(image, method, topk, alpha):
    """Process image and generate Grad-CAM visualizations."""
    if app_state["model"] is None:
        return "❌ No model loaded", [], []
    
    if image is None:
        return "❌ No image provided", [], []
    
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare image
        pil = image.convert("RGB")
        x = app_state["transform"](pil)
        x_batched = x.unsqueeze(0)
        
        # Generate base image for overlay
        base_pil = denorm_to_pil(
            x, 
            app_state["meta"].get("mean", [0.2860]), 
            app_state["meta"].get("std", [0.3530])
        )
        
        # Run prediction and CAM
        device = get_device("cpu")
        cam_results, probs = predict_and_cam(
            app_state["model"], x_batched, device, 
            app_state["target_layer"], topk=topk, method=method
        )
        
        # Create predictions table
        predictions = []
        for r in cam_results:
            class_name = app_state["classes"][r["class_index"]] if app_state["classes"] else str(r["class_index"])
            predictions.append([
                r["rank"],
                class_name,
                r["class_index"],
                f"{r['prob']:.4f}"
            ])
        
        # Create overlay images
        overlays = []
        for r in cam_results:
            class_name = app_state["classes"][r["class_index"]] if app_state["classes"] else str(r["class_index"])
            overlay_img = overlay_pil(base_pil, r["cam"], alpha=alpha)
            overlays.append((overlay_img, f"Top{r['rank']}: {class_name} ({r['prob']:.3f})"))
        
        return "✅ Processing complete", predictions, overlays
    
    except Exception as e:
        return f"❌ Processing failed: {str(e)}", [], []


# Create Gradio interface
def create_interface():
    presets = load_release_presets()
    preset_choices = ["None"] + list(presets.keys()) if presets else ["None"]
    
    with gr.Blocks(css="""
    .alert {
    padding: 10px 15px;
    background-color: #FFF3CD;
    color: #856404;
    border: 1px solid #FFEEBA;
    border-radius: 6px;
    position: relative;
    text-color: #856404;
    }
    """, theme=custom_theme) as demo:
        gr.Markdown("# 🔍 Grad-CAM Demo — Upload an image, get top-k predictions + heatmaps")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Settings")
                
                # Checkpoint loading
                gr.Markdown("### Load Checkpoint")
                with gr.Group():
                    preset_dropdown = gr.Dropdown(
                        choices=preset_choices, 
                        value="None",
                        label="Preset (GitHub Releases)"
                    )
                    url_input = gr.Textbox(
                        label="Or paste asset URL",
                        placeholder="https://github.com/user/repo/releases/download/..."
                    )
                    url_button = gr.Button("Download from URL", variant="primary")
                
                with gr.Group():
                    file_input = gr.File(
                        label="Upload checkpoint (.ckpt)",
                        file_types=[".ckpt"]
                    )
                    file_button = gr.Button("Load uploaded file", variant="primary")
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="No checkpoint loaded"
                )
                
                meta_display = gr.Code(
                    label="Model Metadata",
                    language="json",
                    interactive=False
                )
                
                # Processing options
                gr.Markdown("### Processing Options")
                method_radio = gr.Radio(
                    choices=["Grad-CAM", "Grad-CAM++"],
                    value="Grad-CAM",
                    label="CAM Method"
                )
                topk_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Top-k classes"
                )
                alpha_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Overlay alpha"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## Image Input")

                gr.HTML(
                    """
                    <style>
                    .close-toggle {
                        /* Hide the checkbox itself */
                        position: absolute;
                        opacity: 0;
                        pointer-events: none;
                    }

                    /* When checked, hide the alert */
                    .close-toggle:checked + .alert {
                        display: none;
                    }

                    .alert {
                        position: relative;
                        padding: 12px 40px 12px 12px;
                        background: #fff3cd;   /* pale yellow */
                        color: #664d03;
                        border: 1px solid #ffe69c;
                        border-radius: 8px;
                        font-family: system-ui, sans-serif;
                    }

                    .alert .close {
                        position: absolute;
                        top: 6px;
                        right: 10px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #664d03;
                        cursor: pointer;
                        user-select: none;
                        text-decoration: none;
                    }
                    </style>

                    <input id="alert-close-1" class="close-toggle" type="checkbox">

                    <div class="alert">
                    <label for="alert-close-1" class="close" aria-label="Close alert">&times;</label>
                    ⚠️ Image was resized for better visualization — not equal to dataset original size.
                    </div>
                    """
                )
                
                with gr.Group():

                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=400,
                    )
                    
                    with gr.Row():
                        sample_button = gr.Button("Random Sample", visible=False)
                        
                    with gr.Group():
                        gr.Markdown("**Dataset Sample Browser**")
                        class_filter = gr.Dropdown(
                            label="Filter by class",
                            choices=["(any)"],
                            value="(any)",
                            visible=False
                        )
                        sample_slider = gr.Slider(
                            label="Sample index",
                            minimum=0,
                            maximum=0,
                            value=0,
                            step=1,
                            visible=False,
                            interactive=True
                        )
                        sample_info = gr.Textbox(
                            label="Sample Info",
                            interactive=False,
                            visible=False
                        )
                
                process_button = gr.Button("🔍 Process Image", variant="primary", size="lg")
                process_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False
                )
                
                gr.Markdown("## Results")
                
                with gr.Group():
                    gr.Markdown("### Top-k Predictions")
                    predictions_table = gr.Dataframe(
                        headers=["Rank", "Class", "Index", "Probability"],
                        datatype=["number", "str", "number", "str"],
                        interactive=False
                    )
                
                with gr.Group():
                    gr.Markdown("### Grad-CAM Overlays")
                    overlay_gallery = gr.Gallery(
                        label="CAM Overlays",
                        show_label=False,
                        elem_id="gallery",
                        columns=3,
                        object_fit="contain",
                        height="auto"
                    )
        
        # Event handlers
        url_button.click(
            fn=load_checkpoint_from_url,
            inputs=[url_input, preset_dropdown],
            outputs=[status_text, meta_display, sample_button, class_filter, sample_slider, sample_info]
        )
        
        file_button.click(
            fn=load_checkpoint_from_file,
            inputs=[file_input],
            outputs=[status_text, meta_display, sample_button, class_filter, sample_slider, sample_info]
        )
        
        sample_button.click(
            fn=get_random_sample,
            inputs=[class_filter],
            outputs=[image_input, sample_info, sample_slider]
        )
        
        class_filter.change(
            fn=update_class_filter,
            inputs=[class_filter],
            outputs=[sample_slider]
        )
        
        sample_slider.change(
            fn=get_sample_by_index,
            inputs=[sample_slider, class_filter],
            outputs=[image_input, sample_info]
        )
        
        process_button.click(
            fn=process_image,
            inputs=[image_input, method_radio, topk_slider, alpha_slider],
            outputs=[process_status, predictions_table, overlay_gallery]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )