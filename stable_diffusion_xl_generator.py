#!/usr/bin/env python3
"""
Stable Diffusion XL Image Generation Script for Newton Cluster
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline
from datasets import load_dataset
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')

# Setting Hugging Face token from environment variable
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

from huggingface_hub import login
login(token=HF_TOKEN)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device

def load_prompts(start_idx=220, end_idx=230):
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    
    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test")
    prompts_df = pd.DataFrame(dataset[start_idx:end_idx])
    prompts = prompts_df['Prompt'].tolist()
    
    print(f"\nLoaded prompts #{start_idx+1} to #{end_idx}")
    for i, prompt in enumerate(prompts, start_idx+1):
        print(f"{i}. {prompt[:70]}...")
    
    # Clear dataset from memory
    del dataset, prompts_df
    clear_memory()
    return prompts

def load_model():
    print("\n" + "="*70)
    print("LOADING SDXL MODEL")
    print("="*70)


    pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    
    pipe_sdxl.enable_model_cpu_offload()
    print("Model CPU offload enabled")


    try:
        pipe_sdxl.enable_xformers_memory_efficient_attention()
        print("XFormers enabled")
    except:
        print("Using default attention")

    
    pipe_sdxl.enable_vae_tiling()
    print("VAE tiling enabled")

    print("SDXL loaded with memory optimizations")
    return pipe_sdxl

def generate_sdxl_images(pipe_sdxl, prompts, output_dir="sdxl_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    generation_times = []
    image_paths = []
    
    print("\n" + "="*70)
    print("GENERATING SDXL IMAGES (512x512, 25 steps)")
    print("="*70)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        
        # Clear memory before each generation
        clear_memory()
        
        start_time = time.time()
        
        # Generate image with optimized settings
        image = pipe_sdxl(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.5,
            num_images_per_prompt=1
        ).images[0]
        
        end_time = time.time()
        generation_time = end_time - start_time
        generation_times.append(generation_time)
        
        # Save immediately and don't keep in memory
        image_path = os.path.join(output_dir, f"sdxl_image_{i+1:02d}.png")
        image.save(image_path)
        image_paths.append(image_path)
        
        print(f"Generated in {generation_time:.2f}s - Saved to {image_path}")
        
        # Delete image from memory immediately
        del image
        clear_memory()
    
    return image_paths, generation_times

def create_performance_analysis(sdxl_times, prompts, sdxl_paths):
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    
    # Create results DataFrame
    results = {
        'Metric': ['Total Images', 'Resolution', 'Steps', 'Avg Time (s)',
                   'Min Time (s)', 'Max Time (s)', 'Total Time (s)', 'Total Time (min)'],
        'SDXL': [len(prompts), '1024x1024', 28,
                 f"{np.mean(sdxl_times):.2f}",
                 f"{min(sdxl_times):.2f}",
                 f"{max(sdxl_times):.2f}",
                 f"{sum(sdxl_times):.2f}",
                 f"{sum(sdxl_times)/60:.2f}"]
    }
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    results_df.to_csv('sdxl_performance_results.csv', index=False)
    print("\nSaved: sdxl_performance_results.csv")
    
    # Create detailed timing data
    timing_df = pd.DataFrame({
        'Image_Number': range(1, len(prompts) + 1),
        'Prompt': [p[:50] + '...' for p in prompts],
        'Generation_Time_seconds': sdxl_times
    })
    timing_df.to_csv('sdxl_detailed_timings.csv', index=False)
    print("Saved: sdxl_detailed_timings.csv")
    
    create_visualizations(sdxl_times, prompts, sdxl_paths)

def create_visualizations(sdxl_times, prompts, sdxl_paths):
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time per image
    axes[0, 0].plot(range(1, len(sdxl_times) + 1), sdxl_times,
                    marker='o', linewidth=2.5, markersize=8,
                    color='#3498db', alpha=0.7, label='SDXL')
    axes[0, 0].axhline(y=np.mean(sdxl_times), color='red',
                        linestyle='--', label=f'Avg: {np.mean(sdxl_times):.2f}s')
    axes[0, 0].set_title('Generation Time per Image', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Image Number', fontsize=11)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Statistics bar chart
    stats = ['Avg', 'Min', 'Max']
    values = [np.mean(sdxl_times), min(sdxl_times), max(sdxl_times)]
    axes[0, 1].bar(stats, values, color=['#3498db', '#2ecc71', '#e74c3c'],
                   edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Time Statistics', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=11)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}s', ha='center', fontweight='bold')
    
    # Cumulative time
    cumulative = np.cumsum(sdxl_times)
    axes[1, 0].plot(range(1, len(cumulative) + 1), cumulative / 60,
                    marker='s', linewidth=2.5, markersize=8,
                    color='#9b59b6', alpha=0.7)
    axes[1, 0].set_title('Cumulative Generation Time', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Image Number', fontsize=11)
    axes[1, 0].set_ylabel('Time (minutes)', fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # Summary text
    summary = f"""
MODEL: Stable Diffusion XL

IMAGES GENERATED: {len(prompts)}

RESOLUTION: 1024x1024

INFERENCE STEPS: 28

AVERAGE TIME: {np.mean(sdxl_times):.2f}s

MIN TIME: {min(sdxl_times):.2f}s

MAX TIME: {max(sdxl_times):.2f}s

TOTAL TIME: {sum(sdxl_times):.2f}s
           ({sum(sdxl_times)/60:.2f} minutes)

THROUGHPUT: {3600/np.mean(sdxl_times):.1f} images/hour
"""
    axes[1, 1].text(0.1, 0.5, summary, fontsize=11,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[1, 1].axis('off')
    
    plt.suptitle(f'SDXL Performance Analysis - {len(prompts)} Images',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sdxl_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: sdxl_performance_analysis.png")
    
    # Create sample images grid if images exist
    if os.path.exists('sdxl_images'):
        create_image_grid(sdxl_paths, sdxl_times)

def create_image_grid(sdxl_paths, sdxl_times):
    print("\nCreating sample images grid...")
    
    # Load images
    sdxl_images = []
    for path in sdxl_paths:
        if os.path.exists(path):
            img = Image.open(path)
            sdxl_images.append(img)
    
    if sdxl_images:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(min(len(sdxl_images), 10)):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(sdxl_images[i])
            axes[row, col].set_title(f'SDXL #{i+1}\n{sdxl_times[i]:.2f}s', fontsize=10)
            axes[row, col].axis('off')
        
        plt.suptitle('SDXL Generated Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sdxl_sample_images.png', dpi=300, bbox_inches='tight')
        print("Saved: sdxl_sample_images.png")

def main():
    print("Starting Stable Diffusion XL Image Generation")
    
    # Setup
    device = setup_device()
    
    # Load data
    prompts = load_prompts()
    
    # Load model
    pipe_sdxl = load_model()
    
    # Generate images
    sdxl_paths, sdxl_times = generate_sdxl_images(pipe_sdxl, prompts)
    
    print("\n" + "="*70)
    print("SDXL GENERATION COMPLETE")
    print("="*70)
    print(f"Average time: {np.mean(sdxl_times):.2f}s")
    print(f"Total time: {sum(sdxl_times):.2f}s ({sum(sdxl_times)/60:.2f} min)")
    print(f"Min time: {min(sdxl_times):.2f}s")
    print(f"Max time: {max(sdxl_times):.2f}s")
    print(f"Throughput: {3600/np.mean(sdxl_times):.1f} images/hour")
    
    # Clean up model
    del pipe_sdxl
    clear_memory()
    print("Model removed from memory")
    
    # Wait for memory to clear
    time.sleep(2)
    
    # Create analysis
    create_performance_analysis(sdxl_times, prompts, sdxl_paths)
    
    print("\nAll outputs saved in current directory")
    print("Job completed successfully!")

if __name__ == "__main__":
    main()