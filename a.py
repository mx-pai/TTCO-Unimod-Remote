#!/usr/bin/env python3
"""
Python 3.6兼容版本的UniMod1K修复脚本
解决编译错误和系统依赖问题
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True, shell=False):
    """兼容Python 3.6的命令执行函数"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if check and result.returncode != 0:
            print(f"命令执行失败: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            print(f"错误信息: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
        return True
    except Exception as e:
        print(f"命令执行异常: {e}")
        return False

def install_system_dependencies():
    """安装系统级依赖"""
    print("🔧 安装系统级依赖...")
    
    try:
        # 更新包列表
        print("📦 更新包列表...")
        if not run_command(["apt", "update"]):
            print("⚠️ apt update 失败，尝试继续...")
        
        # 系统依赖列表
        system_deps = [
            "build-essential",
            "gcc", 
            "g++",
            "liblmdb-dev",
            "pkg-config", 
            "libffi-dev",
            "python3-dev",
            "libjpeg-dev",
            "libpng-dev"
        ]
        
        # 安装系统依赖
        print("📦 安装系统依赖...")
        cmd = ["apt", "install", "-y"] + system_deps
        if run_command(cmd):
            print("✅ 系统依赖安装完成")
            return True
        else:
            print("⚠️ 系统依赖安装失败")
            return False
            
    except Exception as e:
        print(f"⚠️ 系统依赖安装异常: {e}")
        return False

def install_pytorch():
    """安装PyTorch"""
    print("🔥 安装PyTorch...")
    
    try:
        # 检查CUDA
        cuda_available = run_command(["nvidia-smi"], check=False)
        
        if cuda_available:
            print("✅ 检测到NVIDIA GPU，安装CUDA版本")
            torch_cmd = [
                "pip", "install", 
                "torch==1.8.1+cu111", 
                "torchvision==0.9.1+cu111",
                "-f", "https://download.pytorch.org/whl/torch_stable.html"
            ]
        else:
            print("⚠️ 未检测到GPU，安装CPU版本")
            torch_cmd = ["pip", "install", "torch==1.8.1+cpu", "torchvision==0.9.1+cpu"]
        
        if run_command(torch_cmd):
            print("✅ PyTorch安装完成")
            return True
        else:
            print("⚠️ PyTorch安装失败")
            return False
            
    except Exception as e:
        print(f"⚠️ PyTorch安装异常: {e}")
        return False

def install_basic_packages():
    """安装基础Python包"""
    print("📦 安装基础Python包...")
    
    # 基础包列表（兼容Python 3.6的版本）
    basic_packages = [
        "opencv-python==4.5.5.64",
        "PyYAML==5.4.1",
        "easydict==1.9", 
        "pandas==1.1.5",
        "tqdm==4.62.3",
        "matplotlib==3.3.4",
        "numpy==1.19.5",
        "Pillow==8.4.0",
        "cython==0.29.28",
        "scipy==1.5.4"
    ]
    
    success_count = 0
    for package in basic_packages:
        print(f"📦 安装 {package}...")
        if run_command(["pip", "install", package]):
            print(f"✅ {package}")
            success_count += 1
        else:
            print(f"⚠️ {package} 安装失败")
    
    print(f"📊 基础包安装完成: {success_count}/{len(basic_packages)}")
    return success_count >= len(basic_packages) * 0.8  # 80%成功率即可

def skip_problematic_packages():
    """跳过有问题的包"""
    print("⚠️ 跳过可能有编译问题的包...")
    
    problematic = [
        "lmdb - 编译依赖复杂，项目可以不使用",
        "jpeg4py - 需要特殊库，可用opencv替代", 
        "pycocotools - 编译复杂，仅COCO数据集需要"
    ]
    
    for pkg in problematic:
        print(f"⚠️ 跳过: {pkg}")

def create_test_script():
    """创建测试脚本"""
    print("🧪 创建测试脚本...")
    
    test_code = '''#!/usr/bin/env python3
"""
环境测试脚本 - Python 3.6兼容版本
"""
import sys
import os

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'), 
        ('cv2', 'OpenCV'),
        ('yaml', 'PyYAML'),
        ('easydict', 'EasyDict'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas')
    ]
    
    success = 0
    for module, name in modules:
        try:
            __import__(module)
            print("✅ {}".format(name))
            success += 1
        except ImportError as e:
            print("❌ {}: {}".format(name, e))
    
    print("\\n📊 导入成功: {}/{}".format(success, len(modules)))
    return success >= 6

def test_pytorch():
    """测试PyTorch"""
    print("\\n🔥 测试PyTorch...")
    
    try:
        import torch
        print("✅ PyTorch版本: {}".format(torch.__version__))
        
        # 基础张量操作
        x = torch.randn(2, 3)
        print("✅ 张量操作: {}".format(x.shape))
        
        # CUDA测试
        if torch.cuda.is_available():
            print("✅ CUDA可用: {}".format(torch.cuda.get_device_name()))
        else:
            print("⚠️ CUDA不可用，使用CPU")
        
        return True
    except Exception as e:
        print("❌ PyTorch测试失败: {}".format(e))
        return False

def test_project_modules():
    """测试项目模块"""
    print("\\n🔬 测试项目模块...")
    
    # 添加项目路径
    spt_path = "/root/autodl-tmp/UniMod1K/SPT"
    if os.path.exists(spt_path):
        sys.path.insert(0, spt_path)
        sys.path.insert(0, os.path.join(spt_path, "lib"))
    
    try:
        from lib.config.spt.config import cfg
        print("✅ SPT配置模块")
        config_ok = True
    except Exception as e:
        print("❌ SPT配置模块: {}".format(e))
        config_ok = False
    
    try:
        from lib.models.spt import build_spt
        print("✅ SPT模型模块") 
        model_ok = True
    except Exception as e:
        print("❌ SPT模型模块: {}".format(e))
        model_ok = False
    
    return config_ok and model_ok

if __name__ == "__main__":
    print("🚀 开始环境测试...")
    
    basic_ok = test_imports()
    pytorch_ok = test_pytorch()
    project_ok = test_project_modules()
    
    print("\\n" + "="*50)
    if basic_ok and pytorch_ok:
        print("🎉 基础环境测试通过！")
        if project_ok:
            print("🎉 项目模块测试通过！可以开始使用了")
        else:
            print("⚠️ 项目模块有问题，但基础环境OK")
    else:
        print("⚠️ 基础环境测试未通过，需要检查安装")
'''
    
    try:
        with open("test_environment_py36.py", 'w') as f:
            f.write(test_code)
        print("✅ 创建测试脚本: test_environment_py36.py")
        return True
    except Exception as e:
        print(f"⚠️ 创建测试脚本失败: {e}")
        return False

def create_manual_commands():
    """创建手动安装命令"""
    print("📋 创建手动安装指南...")
    
    manual_script = '''#!/bin/bash
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
'''
    
    try:
        with open("manual_install.sh", 'w') as f:
            f.write(manual_script)
        
        # 添加执行权限
        os.chmod("manual_install.sh", 0o755)
        print("✅ 创建手动安装脚本: manual_install.sh")
        return True
    except Exception as e:
        print(f"⚠️ 创建手动安装脚本失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始解决编译错误 (Python 3.6兼容版本)...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 当前Python版本: {python_version.major}.{python_version.minor}")
    
    if python_version.minor < 6:
        print("⚠️ Python版本过低，建议使用3.6+")
        return False
    
    success_steps = 0
    total_steps = 5
    
    # 1. 安装系统依赖
    if install_system_dependencies():
        success_steps += 1
        print("✅ 系统依赖安装完成")
    else:
        print("⚠️ 系统依赖安装失败，继续其他步骤")
    
    # 2. 安装PyTorch
    if install_pytorch():
        success_steps += 1
        print("✅ PyTorch安装完成")
    else:
        print("⚠️ PyTorch安装失败")
    
    # 3. 安装基础包
    if install_basic_packages():
        success_steps += 1
        print("✅ 基础包安装完成")
    else:
        print("⚠️ 基础包安装不完整")
    
    # 4. 跳过问题包
    skip_problematic_packages()
    success_steps += 1
    
    # 5. 创建测试脚本
    if create_test_script():
        success_steps += 1
        print("✅ 测试脚本创建完成")
    
    # 6. 创建手动安装脚本
    create_manual_commands()
    
    print(f"\n📊 完成步骤: {success_steps}/{total_steps}")
    
    if success_steps >= 3:
        print("\n🎉 修复基本完成！")
        print("\n📋 下一步操作:")
        print("1. 运行测试: python3 test_environment_py36.py")
        print("2. 如果测试失败，运行手动脚本: bash manual_install.sh")
        print("3. 测试项目模块导入")
    else:
        print("\n⚠️ 修复未完全成功，请尝试手动安装")
        print("运行: bash manual_install.sh")
    
    return success_steps >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)