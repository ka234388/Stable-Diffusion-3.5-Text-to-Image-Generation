#!/bin/bash

echo "=========================================="
echo "Starting SD3.5 Evaluation"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

# Load required modules
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

echo "Modules loaded"
echo ""

# Set environment variables
export HF_TOKEN="your_huggingface_token_here" # add the token
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Check GPU availability
echo "Checking GPU availability:"
nvidia-smi
echo ""

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "sd35_env"; then
    echo "Creating conda environment sd35_env..."
    conda create -n sd35_env python=3.9 -y
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sd35_env

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "All packages installed"
echo ""

# Run the evaluation script
echo "=========================================="
echo "Running stable_diffusion_3_5_generator.py"
echo "=========================================="
python stable_diffusion_3_5_generator.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Script completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Script failed with errors"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

# Deactivate conda environment
conda deactivate