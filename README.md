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

## Requirements

```
torch
diffusers
transformers
accelerate
```

## Usage

```python
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="your prompt here",
    num_inference_steps=28,
    height=1024,
    width=1024
).images[0]
```

## Conclusion

SDXL is recommended when generation speed is critical, while SD3.5 Medium is preferred for applications requiring higher visual fidelity. The speed difference is attributed to architectural differences between the models rather than inference parameters.

## Author

**Karthika Ramasamy**

## License

This project is for educational purposes.
