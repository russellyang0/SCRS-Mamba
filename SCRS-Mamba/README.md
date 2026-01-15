# SCRS-Mamba

Official implementation of **SCRS-Mamba** for remote sensing scene classification.

- **Author:** Zaichun Yang

## Overview
SCRS-Mamba introduces a Scale-aware State Space Model (SA-SSM) with spatially continuous multi-view scanning for robust remote sensing scene recognition.

This repository provides:
- Training and evaluation scripts based on MMEngine/MMPretrain
- Model and dataset definitions for SCRS-Mamba
- Feature visualization utilities (Fine/Coarse/Fusion)

## Environment
This codebase is built on the OpenMMLab ecosystem.

### 1) Create environment
```bash
conda create -n scrsmamba python=3.10 -y
conda activate scrsmamba
```

### 2) Install PyTorch
Please install PyTorch following the official instructions for your CUDA version.

### 3) Install OpenMMLab dependencies
Recommended (with OpenMIM):
```bash
pip install -U openmim
mim install "mmcv>=2.0.0,<2.4.0"
pip install "mmengine>=0.8.3,<1.0.0" "mmpretrain>=1.2.0"
```

### 4) Install SCRS-Mamba
```bash
pip install -e .
```

### 5) Install Mamba dependencies
```bash
pip install "transformers>=4.39.0"
pip install mamba-ssm causal-conv1d
```

## Data Preparation
This repository follows the common folder structure:

```text
data/
  AID/
  UCMerced_LandUse/
  NWPU-RESISC45/
```

Prepare train/val splits as plain text lists (relative paths) under `datainfo/`.

## Training
Example: AID (Base) with SA-SSM enabled.

```bash
python tools/train.py configs/scrsmamba/scrsmamba_aid_b_sa_ssm.py --amp
```

Checkpoints and logs will be saved to `work_dirs/`.

## Evaluation
```bash
python tools/test.py configs/scrsmamba/scrsmamba_aid_b_sa_ssm.py \
  work_dirs/scrsmamba_aid_b_sa_ssm/best_*.pth
```

## Feature Visualization (Fine/Coarse/Fusion)
```bash
python tools/visualization/vis_sa_ssm_features.py \
  --config configs/scrsmamba/scrsmamba_aid_b_sa_ssm.py \
  --checkpoint work_dirs/scrsmamba_aid_b_sa_ssm/best_*.pth \
  --images path/to/image1.jpg path/to/image2.jpg \
  --out-dir outputs/sa_vis \
  --img-size 224
```

## Citation
If you find this work useful, please cite:

```bibtex
@unpublished{yang2026scrsmamba,
  title  = {SCRS-Mamba: Scale-aware and Spatially Continuous Multi-View Scanning Mamba for Remote Sensing Scene Classification},
  author = {Yang, Zaichun},
  year   = {2026},
  note   = {Manuscript under review at iScience}
}

```

## Acknowledgements
This project is built upon the OpenMMLab ecosystem (MMEngine/MMPretrain) and related open-source efforts.
