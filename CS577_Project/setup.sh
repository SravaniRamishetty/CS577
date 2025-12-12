#!/bin/bash

set -e

echo "========================================="
echo "Reasoning Direction Analysis Setup"
echo "========================================="

# Check Python version
version_ge() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
MIN_VERSION="3.10.0"

if version_ge $MIN_VERSION $PYTHON_VERSION; then
    echo "Error: Python 3.10 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Create .env file if it doesn't exist
touch .env

# Hugging Face Token Setup
echo ""
echo "========================================="
echo "Hugging Face Setup"
echo "========================================="
read -p "Do you want to set up Hugging Face authentication? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face token: " hf_token
    if [ ! -z "$hf_token" ]; then
        echo "HF_TOKEN=$hf_token" >> .env
        echo "✓ Hugging Face token saved to .env"

        # Install and authenticate with huggingface-cli
        pip install --upgrade huggingface_hub
        echo $hf_token | huggingface-cli login --token $hf_token
        echo "✓ Authenticated with Hugging Face"
    fi
fi

# Weights & Biases Setup (optional)
echo ""
echo "========================================="
echo "Weights & Biases Setup (Optional)"
echo "========================================="
read -p "Do you want to set up Weights & Biases for experiment tracking? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your W&B API key: " wandb_key
    if [ ! -z "$wandb_key" ]; then
        echo "WANDB_API_KEY=$wandb_key" >> .env
        echo "✓ W&B API key saved to .env"
    fi
fi

# Create virtual environment
echo ""
echo "========================================="
echo "Creating Virtual Environment"
echo "========================================="

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "========================================="
echo "Upgrading pip"
echo "========================================="
pip install --upgrade pip

# Install requirements
echo ""
echo "========================================="
echo "Installing Dependencies"
echo "========================================="
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "========================================="
echo "Creating Project Directories"
echo "========================================="
mkdir -p data/gsm8k
mkdir -p data/math
mkdir -p results/activations
mkdir -p results/directions
mkdir -p results/interventions
mkdir -p results/probes
mkdir -p results/visualizations
mkdir -p pipeline/runs
mkdir -p notebooks

echo "✓ Project directories created"

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo ""
    echo "========================================="
    echo "Initializing Git Repository"
    echo "========================================="
    git init
    echo "✓ Git repository initialized"
fi

# Create .gitignore
echo ""
echo "========================================="
echo "Creating .gitignore"
echo "========================================="
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env
.env.local

# Data and Models
data/*/
!data/.gitkeep
models/
*.pt
*.pth
*.bin
*.safetensors

# Results
results/*/
!results/.gitkeep
pipeline/runs/*/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Weights & Biases
wandb/

# Cache
.cache/
*.cache
EOF

echo "✓ .gitignore created"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the project, see README.md for instructions."
echo ""
