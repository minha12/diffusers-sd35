# Diffusers Environment Setup on Berzelius

This guide explains how to set up the required environment for running diffusers on Berzelius HPC.

> **Note**: As of March 1, 2024, Berzelius supports CUDA 12.2. The following instructions are specifically tailored for compatibility with Python 3.10 and PyTorch 2.5.1 to ensure stable performance.

## Prerequisites

- Access to Berzelius HPC
- Loaded Miniforge3 module

## Environment Setup

1. Load the required module:
```bash
module load Miniforge3
```

2. Create a new conda environment:
```bash
mamba create -n diffusers python=3.10 -y
```

3. Activate the environment:
```bash
mamba activate diffusers
```

4. Install PyTorch and related packages:
```bash
mamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

5. Install diffusers:
```bash
mamba install -c conda-forge diffusers
```

## Usage

After setting up the environment, you can run the training script using:
```bash
sbatch run_train_drsk.sh
```