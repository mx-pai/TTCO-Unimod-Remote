#!/bin/bash
#===============================================================================
# SPT Improved Training - One-Click Launch Script
#===============================================================================
# This script:
# 1. Checks environment and dependencies
# 2. Validates paths and configs
# 3. Starts training with optimal settings
# 4. Monitors GPU usage and training progress
#
# Usage:
#   bash launch_training.sh [--config CONFIG_NAME] [--gpus N] [--data-root PATH]
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
CONFIG="unimod1k_improved"
NUM_GPUS=1
KEEP_CKPT=5
SAVE_DIR="./checkpoints_improved"
DATA_ROOT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --keep-ckpt)
            KEEP_CKPT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}SPT IMPROVED TRAINING LAUNCHER${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

#===============================================================================
# 1. Environment Check
#===============================================================================
echo -e "${YELLOW}[1/6] Checking environment...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}✗ CUDA not available!${NC}"
    exit 1
fi
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo -e "${GREEN}✓ CUDA available: ${GPU_COUNT} GPU(s)${NC}"

# Check if requested GPUs <= available
if [ $NUM_GPUS -gt $GPU_COUNT ]; then
    echo -e "${RED}✗ Requested ${NUM_GPUS} GPUs but only ${GPU_COUNT} available${NC}"
    exit 1
fi

# Get GPU info
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo -e "${GREEN}✓ GPU: ${GPU_NAME}${NC}"

#===============================================================================
# 2. Dependency Check
#===============================================================================
echo -e "\n${YELLOW}[2/6] Checking dependencies...${NC}"

REQUIRED_PACKAGES=("torch" "torchvision" "pytorch_pretrained_bert" "tensorboard" "numpy" "cv2" "yaml")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}Install with: pip install ${MISSING_PACKAGES[*]}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies installed${NC}"

#===============================================================================
# 3. Path Validation
#===============================================================================
echo -e "\n${YELLOW}[3/6] Validating paths...${NC}"

# Project root
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ Project root: ${PROJECT_ROOT}${NC}"

# Config file
CONFIG_FILE="experiments/spt/${CONFIG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config not found: ${CONFIG_FILE}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file: ${CONFIG_FILE}${NC}"

# Check critical paths in config
echo -e "${YELLOW}  Checking paths in config...${NC}"
python << EOF
import yaml
import os
import sys

with open('${CONFIG_FILE}', 'r') as f:
    cfg = yaml.safe_load(f)

errors = []

# Check pretrained model
pretrained = cfg.get('MODEL', {}).get('PRETRAINED', '')
if pretrained and not os.path.exists(pretrained):
    if '/path/to/' not in pretrained:  # Not a placeholder
        errors.append(f"Pretrained model not found: {pretrained}")

# Check BERT paths
bert_path = cfg.get('MODEL', {}).get('LANGUAGE', {}).get('PATH', '')
if bert_path and not os.path.exists(bert_path):
    if '/path/to/' not in bert_path:
        errors.append(f"BERT model not found: {bert_path}")

vocab_path = cfg.get('MODEL', {}).get('LANGUAGE', {}).get('VOCAB_PATH', '')
if vocab_path and not os.path.exists(vocab_path):
    if '/path/to/' not in vocab_path:
        errors.append(f"BERT vocab not found: {vocab_path}")

if errors:
    print('\n'.join(errors))
    sys.exit(1)
else:
    print("✓ All paths in config are valid (or placeholders)")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Invalid paths in config${NC}"
    echo -e "${YELLOW}  Please update paths in ${CONFIG_FILE}${NC}"
    exit 1
fi

# Check local.py
if [ ! -f "lib/train/admin/local.py" ]; then
    echo -e "${YELLOW}✗ lib/train/admin/local.py not found${NC}"
    echo -e "${YELLOW}  Creating default local.py...${NC}"
    mkdir -p lib/train/admin
    cat > lib/train/admin/local.py << 'LOCALPY'
from lib.train.admin.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.workspace_dir = '/root/autodl-tmp/UniMod1K/SPT'
    settings.tensorboard_dir = '/root/autodl-tmp/UniMod1K/SPT/tensorboard'
    settings.pretrained_networks = '/root/autodl-tmp'
    settings.unimod1k_dir = '/root/autodl-tmp/1-训练验证集/TrainSet'
    settings.unimod1k_dir_nlp = '/root/autodl-tmp/1-训练验证集/TrainSet'
    return settings
LOCALPY
    echo -e "${GREEN}✓ Created lib/train/admin/local.py${NC}"
fi

if [ ! -f "lib/test/evaluation/local.py" ]; then
    echo -e "${YELLOW}✗ lib/test/evaluation/local.py not found${NC}"
    echo -e "${YELLOW}  Creating default local.py...${NC}"
    mkdir -p lib/test/evaluation
    cat > lib/test/evaluation/local.py << 'TESTLOCALPY'
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = '/root/autodl-tmp/UniMod1K/SPT'
    settings.save_dir = '/root/autodl-tmp/UniMod1K/SPT'
    settings.unimod1k_path = '/root/autodl-tmp/1-训练验证集/ValidationSet'
    settings.results_path = '/root/autodl-tmp/UniMod1K/SPT/lib/test/tracking_results'
    return settings
TESTLOCALPY
    echo -e "${GREEN}✓ Created lib/test/evaluation/local.py${NC}"
fi

#===============================================================================
# 4. Create directories
#===============================================================================
echo -e "\n${YELLOW}[4/6] Creating directories...${NC}"

mkdir -p "$SAVE_DIR"
mkdir -p logs
mkdir -p tensorboard

echo -e "${GREEN}✓ Directories ready${NC}"

#===============================================================================
# 5. Set environment variables
#===============================================================================
echo -e "\n${YELLOW}[5/6] Setting environment...${NC}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

echo -e "${GREEN}✓ PYTHONPATH=${PYTHONPATH}${NC}"
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

#===============================================================================
# 6. Launch training
#===============================================================================
echo -e "\n${YELLOW}[6/6] Launching training...${NC}"
echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}TRAINING CONFIGURATION${NC}"
echo -e "${GREEN}=================================${NC}"
echo -e "Config: ${CONFIG}"
echo -e "Save dir: ${SAVE_DIR}"
if [ -n "$DATA_ROOT" ]; then
    echo -e "Data root: ${DATA_ROOT}"
fi
echo -e "Keep checkpoints: ${KEEP_CKPT}"
echo -e "GPUs: ${NUM_GPUS}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Build command
CMD="python train.py --config ${CONFIG} --save-dir ${SAVE_DIR} --keep-checkpoints ${KEEP_CKPT}"
if [ -n "$DATA_ROOT" ]; then
    CMD="${CMD} --data-root ${DATA_ROOT}"
fi

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${CONFIG}_${TIMESTAMP}.log"

echo -e "${YELLOW}Command: ${CMD}${NC}"
echo -e "${YELLOW}Log file: ${LOG_FILE}${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop training${NC}"
echo ""

# Run training with tee to show output and save to log
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}TRAINING COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo -e "Checkpoints: ${SAVE_DIR}"
    echo -e "Log: ${LOG_FILE}"
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}TRAINING FAILED${NC}"
    echo -e "${RED}=================================${NC}"
    echo -e "Check log: ${LOG_FILE}"
    exit 1
fi
