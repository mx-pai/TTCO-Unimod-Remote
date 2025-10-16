#!/bin/bash
# UniMod1K 手动安装脚本 - Python 3.6兼容版本

echo "🚀 开始手动安装..."

# 1. 系统依赖
echo "📦 安装系统依赖..."
sudo apt update
sudo apt install -y build-essential gcc g++ liblmdb-dev libffi-dev python3-dev

# 2. 激活环境
echo "🔧 激活conda环境..."
source ~/miniconda3/bin/activate spt

# 3. 清理pip缓存
echo "🧹 清理pip缓存..."
pip cache purge

# 4. 安装PyTorch
echo "🔥 安装PyTorch..."
pip install torch==1.8.1 torchvision==0.9.1

# 5. 安装基础包
echo "📦 安装基础包..."
pip install opencv-python PyYAML easydict numpy matplotlib pandas tqdm Pillow cython scipy

# 6. 测试安装
echo "🧪 测试安装..."
python3 -c "import torch, cv2, yaml, numpy; print('✅ 核心模块导入成功')"

echo "✅ 手动安装完成!"
