# Explainable-CNN

End-to-end image classifier with Grad-CAM.

## use defaults from configs/baseline.yaml
python -m src.train --config configs/baseline.yaml

## override a couple of values from CLI
python -m src.train --config configs/baseline.yaml --epochs 12 --lr 5e-4

## switch dataset via CLI
python -m src.train --config configs/baseline.yaml --dataset mnist
