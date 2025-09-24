---
title: ExplainableCNN
---
# ExplainableCNN

End‑to‑end image classification with explainability. Train CNNs on common vision datasets, save checkpoints and metrics, and visualize Grad‑CAM/Grad‑CAM++ heatmaps in a Streamlit or Gradio app.

>**Online App**: You can try the app online at `https://huggingface.co/spaces/Stefano01/ExplainableCNN`

Contents
- Quick start
- Installation
- Datasets
- Training
- Configuration reference
- Streamlit Grad‑CAM demo
- Checkpoints and outputs
- Project layout
- FAQ / Tips

## Quick start

1) Install dependencies (CPU‑only by default):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Train with defaults (Fashion‑MNIST, small CNN):
```bash
python -m src.train --config configs/baseline.yaml
```

3) Launch the Grad‑CAM demo and visualize predictions:
```bash
streamlit run app/streamlit_app.py
```

Or run the Gradio app locally:
```bash
python app/gradio_app.py
```

## Installation

This repo ships with CPU‑only PyTorch wheels via the official extra index in `requirements.txt`. If you have CUDA, you can install the matching GPU wheels from PyTorch and keep the rest of the requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GPU installation (recommended for training)

1) Install CUDA‑enabled PyTorch that matches your driver and CUDA version (see `https://pytorch.org/get-started/locally/`). Examples:

```bash
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# CUDA 11.8
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

2) Install the rest of the project dependencies (excluding torch*):

```bash
pip install -r requirements-gpu.txt
```

Notes
- If you want GPU builds: follow the selector at `https://pytorch.org/get-started/locally/` and install the torch/torchvision/torchaudio triplet before installing the rest of the requirements.
- This project uses: torch/torchvision, torchmetrics, captum/torchcam, lightning (as the newer package name), albumentations, TensorBoard, Streamlit, PyYAML, etc.

## Datasets

Built‑in dataset options for training:
- `fashion-mnist` (default)
- `mnist`
- `cifar10`

Where data lives
- By default datasets are downloaded under `data/`.

## Training

Run with a YAML config
```bash
python -m src.train --config configs/baseline.yaml
```

Override config values from the CLI
```bash
python -m src.train --config configs/baseline.yaml --epochs 12 --lr 5e-4
```

Switch dataset from the CLI
```bash
python -m src.train --config configs/baseline.yaml --dataset mnist
```

Use a ResNet‑18 backbone for CIFAR‑10 (adapted conv1/no maxpool)
```bash
python -m src.train --config configs/cifar10_resnet18.yaml
```


Training flow (high level)
- Loads YAML config and merges CLI overrides
- Builds dataloaders with dataset‑specific transforms and normalization
- Builds model: `smallcnn`, `resnet18_cifar`, or `resnet18_imagenet`
- Optimizer: Adam (default) or SGD with momentum
- Trains with early stopping and ReduceLROnPlateau on val loss
- Writes TensorBoard logs, metrics JSONs, and image reports (confusion matrix)
- Saves `last.ckpt` and `best.ckpt` with model weights and metadata

Outputs per run (under roots from config)
- `runs/<run_id>/` TensorBoard logs
- `checkpoints/<run_id>/last.ckpt` and `best.ckpt`
- `reports/<run_id>/config_effective.yaml`, `metrics.json`, and `figures/confusion_matrix.png`

## Configuration reference

See examples in `configs/`:
- `baseline.yaml` (Fashion‑MNIST + `smallcnn`)
- `cifar10_resnet18.yaml` (CIFAR‑10 + adapted ResNet‑18)

Common keys
- `dataset`: one of `fashion-mnist`, `mnist`, `cifar10`
- `model_name`: `smallcnn` | `resnet18_cifar` | `resnet18_imagenet`
- `data_dir`: root folder for data (default `./data`)
- `batch_size`, `epochs`, `lr`, `weight_decay`, `num_workers`, `seed`, `device`
- `img_size`, `mean`, `std`: image shape and normalization stats
- `optimizer`: `adam` (default) or `sgd`; `momentum` used for SGD
- `log_root`, `ckpt_root`, `reports_root`: base folders for artifacts
- `early_stop`: `{ monitor: val_loss|val_acc, mode: min|max, patience, min_delta }`

CLI flags can override the YAML. For example `--dataset`, `--epochs`, `--lr`, `--model-name`.

## Streamlit Grad‑CAM demo

Start the app
```bash
streamlit run app/streamlit_app.py
```

What it does
- Load a trained checkpoint (`.ckpt`)
- Upload an image or sample one from the corresponding dataset
- Run inference and display top‑k predictions
- Visualize Grad‑CAM or Grad‑CAM++ overlays with adjustable alpha

## Gradio Grad‑CAM demo

Start the Gradio app locally (or try it online at `https://huggingface.co/spaces/Stefano01/ExplainableCNN`):
```bash
python app/gradio_app.py
```

Notes
- The `app.py` file is used only to start the app online.
- Presets, URL download, and checkpoint upload are supported similarly to the Streamlit app.

Devices and CAM methods
- Device: `auto` (default), `cuda`, or `cpu`
- CAM: `Grad-CAM` or `Grad-CAM++` via `torchcam`

Checkpoint metadata expected
- `meta`: `{ dataset, model_name, img_size, mean, std, default_target_layer }`
- `classes`: list of class names (used to label predictions)

## Checkpoints and outputs

Each run writes:
- Checkpoints: `<ckpt_root>/<run_id>/{last.ckpt,best.ckpt}`
- Logs: `<log_root>/<run_id>/` (TensorBoard)
- Reports: `<reports_root>/<run_id>/metrics.json`, `figures/confusion_matrix.png`

Best checkpoint selection respects early‑stopping monitor (`val_loss` or `val_acc`).

## License and acknowledgements

- Uses `torchcam` for CAM extraction and `captum` as a general explainability dependency
- TorchVision models and datasets are used for baselines and data handling

___

If you run into issues, please open an issue with your command, config file, and environment details.
