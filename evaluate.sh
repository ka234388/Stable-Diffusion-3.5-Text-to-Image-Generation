# Stable Diffusion 3.5 Text-to-Image Generation

A performance evaluation comparing **Stable Diffusion 3.5 Medium** and **Stable Diffusion XL (SDXL)** for text-to-image generation tasks.

## Overview

This project benchmarks the generation speed and quality of SD3.5 Medium against SDXL using identical prompts, resolution, and inference parameters to ensure a fair comparison.

## Dataset

- **Source:** [Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
- **Split:** Test set
- **Prompts Used:** 10 text prompts

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 × 1024 |
| Inference Steps | 28 |
| Total Images | 10 |

### Hardware Configurations Tested

1. **Tesla T4** - 15.83 GB Memory
2. **Tesla V100-PCIE-16GB** - 16.93 GB Memory
3. **NVIDIA A100-SXM4-80GB** - 85.17 GB Memory (SDXL baseline)

## Results

### Performance Comparison (Tesla V100-PCIE-16GB)

| Model | Avg Time (s) | Total Time (s) | Min Time (s) | Max Time (s) |
|-------|--------------|----------------|--------------|--------------|
| SDXL | 13.07 | 130.70 | 12.41 | 17.70 |
| SD3.5 | 25.28 | 252.78 | 20.53 | 66.33 |

### Performance Comparison (Tesla T4 vs A100)

| Model | Avg Time (s) | Total Time (s) | Min Time (s) | Max Time (s) |
|-------|--------------|----------------|--------------|--------------|
| SDXL (A100) | 25.42 | 254.17 | 23.20 | 26.62 |
| SD3.5 (T4) | 65.84 | 658.44 | 62.95 | 88.69 |

## Key Findings

- **Speed:** SDXL generates images approximately **2-2.5x faster** than SD3.5 Medium under identical conditions
- **Quality:** SD3.5 Medium produces slightly higher-quality images with improved detail and visual consistency
- **Use Cases:**
  - **SD3.5 Medium** → Tasks prioritizing image realism and fine details
  - **SDXL** → Real-time applications, rapid prototyping, and large-scale generation

## Prerequisites

- Python 3.9
- CUDA 11.8
- cuDNN 8.6
- Conda package manager
- Hugging Face account with access token
- NVIDIA GPU (Tesla T4, V100, or A100 recommended)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ka234388/Stable-Diffusion-3.5-Text-to-Image-Generation.git
cd Stable-Diffusion-3.5-Text-to-Image-Generation
```

### 2. Configure Hugging Face Token

Open `evaluate.sh` and replace the placeholder with your Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

> **Note:** You need to request access to the SD3.5 model on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) before running.

### 3. Run the Evaluation

Make the script executable and run:

```bash
chmod +x evaluate.sh
./evaluate.sh
```

## What the Script Does

The `evaluate.sh` script automates the entire setup and execution:

1. **Loads required modules** (Python 3.9, CUDA 11.8, cuDNN 8.6)
2. **Checks GPU availability** via `nvidia-smi`
3. **Creates conda environment** (`sd35_env`) if it doesn't exist
4. **Installs PyTorch** with CUDA support
5. **Installs dependencies** from `requirements.txt`
6. **Runs the evaluation** script `stable_diffusion_3_5_generator.py`

## Project Structure

```
├── evaluate.sh                        # Main execution script
├── stable_diffusion_3_5_generator.py  # Image generation & benchmarking code
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Requirements

The following packages are installed automatically via `requirements.txt`:

```
torch
torchvision
torchaudio
diffusers
transformers
accelerate
datasets
```

## Conclusion

SDXL is recommended when generation speed is critical, while SD3.5 Medium is preferred for applications requiring higher visual fidelity. The speed difference is attributed to architectural differences between the models rather than inference parameters.

## Author

**Karthika Ramasamy**
This project is for educational purposes.
