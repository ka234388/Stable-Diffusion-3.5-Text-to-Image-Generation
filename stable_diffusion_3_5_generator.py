#!/usr/bin/env python3
"""
Stable Diffusion 3.5 Image Generation Script for Newton Cluster
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline
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
    import os, torch
    from diffusers import StableDiffusion3Pipeline

    print("\n" + "="*70)
    print("LOADING STABLE DIFFUSION 3.5 MEDIUM MODEL")
    print("="*70)

    pipe_sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        low_cpu_mem_usage=False,
        device_map=None,
        text_encoder_3=None,   
        tokenizer_3=None   
    )

    try:
        pipe_sd35.enable_xformers_memory_efficient_attention()
        print(" XFormers enabled")
    except Exception as e:
        print(f" XFormers not enabled: {e}")

    try:
        pipe_sd35.vae.enable_slicing()
        pipe_sd35.vae.enable_tiling()
        print("VAE slicing/tiling enabled")
    except Exception as e:
        print(f" VAE optimizations not enabled: {e}")

    try:
        pipe_sd35.enable_attention_slicing("max")
        print(" Attention slicing enabled")
    except Exception as e:
        print(f" Attention slicing not enabled: {e}")

    pipe_sd35.enable_model_cpu_offload()
    print(" Model CPU offload enabled (prevents OOM)")

    print("SD3.5 Medium loaded successfully with memory optimizations")
    return pipe_sd35


    print("SD3.5 Medium loaded with memory optimizations (no xFormers)")
    return pipe_sd35



def generate_sd35_images(pipe_sd35, prompts, output_dir="sd35_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    generation_times = []
    image_paths = []
    
    print("\n" + "="*70)
    print("GENERATING SD3.5 IMAGES (1024x1024, 28 steps)")
    print("="*70)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        
        clear_memory()
        start_time = time.time()
        
        image = pipe_sd35(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
            num_images_per_prompt=1,
            max_sequence_length=256 

        ).images[0]
        
        end_time = time.time()
        generation_time = end_time - start_time
        generation_times.append(generation_time)
        
        image_path = os.path.join(output_dir, f"sd35_image_{i+1:02d}.png")
        image.save(image_path)
        image_paths.append(image_path)
        
        print(f"Generated in {generation_time:.2f}s - Saved to {image_path}")
        
        del image
        clear_memory()
    
    return image_paths, generation_times

def create_performance_analysis(sd35_times, prompts, sd35_paths):
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    
    # Create results DataFrame
    results = {
        'Metric': ['Total Images', 'Resolution', 'Steps', 'Avg Time (s)',
                   'Min Time (s)', 'Max Time (s)', 'Total Time (s)', 'Total Time (min)'],
        'SD3.5 Medium': [len(prompts), '1024x1024', 28,
                         f"{np.mean(sd35_times):.2f}",
                         f"{min(sd35_times):.2f}",
                         f"{max(sd35_times):.2f}",
                         f"{sum(sd35_times):.2f}",
                         f"{sum(sd35_times)/60:.2f}"]
    }
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    results_df.to_csv('sd35_performance_results.csv', index=False)
    print("\nSaved: sd35_performance_results.csv")
    
    # Create detailed timing data
    timing_df = pd.DataFrame({
        'Image_Number': range(1, len(prompts) + 1),
        'Prompt': [p[:50] + '...' for p in prompts],
        'Generation_Time_seconds': sd35_times
    })
    timing_df.to_csv('sd35_detailed_timings.csv', index=False)
    print("Saved: sd35_detailed_timings.csv")
    
    create_visualizations(sd35_times, prompts, sd35_paths)

def create_visualizations(sd35_times, prompts, sd35_paths):
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time per image
    axes[0, 0].plot(range(1, len(sd35_times) + 1), sd35_times,
                    marker='o', linewidth=2.5, markersize=8,
                    color='#2ecc71', alpha=0.7, label='SD3.5')
    axes[0, 0].axhline(y=np.mean(sd35_times), color='red',
                        linestyle='--', label=f'Avg: {np.mean(sd35_times):.2f}s')
    axes[0, 0].set_title('Generation Time per Image', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Image Number', fontsize=11)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    stats = ['Avg', 'Min', 'Max']
    values = [np.mean(sd35_times), min(sd35_times), max(sd35_times)]
    axes[0, 1].bar(stats, values, color=['#3498db', '#2ecc71', '#e74c3c'],
                   edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Time Statistics', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=11)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}s', ha='center', fontweight='bold')
    
    cumulative = np.cumsum(sd35_times)
    axes[1, 0].plot(range(1, len(cumulative) + 1), cumulative / 60,
                    marker='s', linewidth=2.5, markersize=8,
                    color='#9b59b6', alpha=0.7)
    axes[1, 0].set_title('Cumulative Generation Time', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Image Number', fontsize=11)
    axes[1, 0].set_ylabel('Time (minutes)', fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # Summary text
    summary = f"""
MODEL: Stable Diffusion 3.5 Medium

IMAGES GENERATED: {len(prompts)}

RESOLUTION: 1024x1024

INFERENCE STEPS: 28

AVERAGE TIME: {np.mean(sd35_times):.2f}s

MIN TIME: {min(sd35_times):.2f}s

MAX TIME: {max(sd35_times):.2f}s

TOTAL TIME: {sum(sd35_times):.2f}s
           ({sum(sd35_times)/60:.2f} minutes)

THROUGHPUT: {3600/np.mean(sd35_times):.1f} images/hour
"""
    axes[1, 1].text(0.1, 0.5, summary, fontsize=11,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[1, 1].axis('off')
    
    plt.suptitle(f'SD3.5 Performance Analysis - {len(prompts)} Images',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sd35_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: sd35_performance_analysis.png")
    
    # Create sample images grid if images exist
    if os.path.exists('sd35_images'):
        create_image_grid(sd35_paths, sd35_times)

def create_image_grid(sd35_paths, sd35_times):
    print("\nCreating sample images grid...")
    
    # Load images
    sd35_images = []
    for path in sd35_paths:
        if os.path.exists(path):
            img = Image.open(path)
            sd35_images.append(img)
    
    if sd35_images:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(min(len(sd35_images), 10)):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(sd35_images[i])
            axes[row, col].set_title(f'SD3.5 #{i+1}\n{sd35_times[i]:.2f}s', fontsize=10)
            axes[row, col].axis('off')
        
        plt.suptitle('SD3.5 Generated Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sd35_sample_images.png', dpi=300, bbox_inches='tight')
        print("Saved: sd35_sample_images.png")

def main():
    print("Starting Stable Diffusion 3.5 Image Generation")
    
    # Setup
    device = setup_device()
    
    # Load data
    prompts = load_prompts()
    
    # Load model
    pipe_sd35 = load_model()
    
    # Generate images
    sd35_paths, sd35_times = generate_sd35_images(pipe_sd35, prompts)
    
    print("\n" + "="*70)
    print("SD3.5 GENERATION COMPLETE")
    print("="*70)
    print(f"Average time: {np.mean(sd35_times):.2f}s")
    print(f"Total time: {sum(sd35_times):.2f}s ({sum(sd35_times)/60:.2f} min)")
    print(f"Min time: {min(sd35_times):.2f}s")
    print(f"Max time: {max(sd35_times):.2f}s")
    
    # Clean up model
    del pipe_sd35
    clear_memory()
    print("Model removed from memory")
    
    # Create analysis
    create_performance_analysis(sd35_times, prompts, sd35_paths)
    
    print("\nAll outputs saved in current directory")
    print("Job completed successfully!")

if __name__ == "__main__":
    main()
