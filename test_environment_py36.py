#!/usr/bin/env python3
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
    
    print("\n📊 导入成功: {}/{}".format(success, len(modules)))
    return success >= 6

def test_pytorch():
    """测试PyTorch"""
    print("\n🔥 测试PyTorch...")
    
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
    print("\n🔬 测试项目模块...")
    
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
    
    print("\n" + "="*50)
    if basic_ok and pytorch_ok:
        print("🎉 基础环境测试通过！")
        if project_ok:
            print("🎉 项目模块测试通过！可以开始使用了")
        else:
            print("⚠️ 项目模块有问题，但基础环境OK")
    else:
        print("⚠️ 基础环境测试未通过，需要检查安装")
