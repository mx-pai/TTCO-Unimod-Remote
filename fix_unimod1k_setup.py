#!/usr/bin/env python3
"""
UniMod1K项目配置修复脚本 v2.0
解决常见的路径、依赖和配置问题
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import shutil

class UniMod1KFixer:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        # 修复：检查实际的目录结构
        if (self.project_root / "SPT").exists():
            self.spt_root = self.project_root / "SPT"
        elif (self.project_root / "UniMod1K" / "SPT").exists():
            self.spt_root = self.project_root / "UniMod1K" / "SPT"
        else:
            # 如果在当前目录，假设就是SPT目录
            self.spt_root = self.project_root

        print(f"🔍 项目根目录: {self.project_root}")
        print(f"🔍 SPT目录: {self.spt_root}")

    def fix_environment(self):
        """修复环境配置"""
        print("🔧 修复环境配置...")

        # 检查Python版本
        python_version = sys.version_info
        print(f"当前Python版本: {python_version.major}.{python_version.minor}")

        if python_version.major == 3 and python_version.minor >= 6:
            print("✅ Python版本兼容")
        else:
            print("⚠️ 建议使用Python 3.6+")

        # 确保SPT目录存在
        os.makedirs(self.spt_root, exist_ok=True)

        # 创建现代化的requirements.txt
        requirements = [
            "torch>=1.7.0,<=1.10.0",  # 兼容原项目
            "torchvision>=0.8.0,<=0.11.0",
            "opencv-python",
            "PyYAML",
            "easydict",
            "cython",
            "pandas",
            "tqdm",
            "pycocotools",
            "tensorboard",
            "matplotlib",
            "numpy",
            "Pillow",
            "transformers>=4.0.0",  # 用于BERT
            "timm",  # 用于视觉模型
            "yacs",  # 配置管理
        ]

        requirements_file = self.spt_root / "requirements.txt"
        try:
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(requirements))
            print(f"✅ 创建requirements.txt: {requirements_file}")
        except Exception as e:
            print(f"⚠️ 创建requirements.txt失败: {e}")

    def fix_paths(self, data_dir=None, pretrained_dir=None):
        """修复路径配置"""
        print("🔧 修复路径配置...")

        # 默认路径
        if data_dir is None:
            data_dir = self.project_root / "data"
        else:
            data_dir = Path(data_dir)

        if pretrained_dir is None:
            pretrained_dir = self.project_root / "pretrained_models"
        else:
            pretrained_dir = Path(pretrained_dir)

        # 创建必要的目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(pretrained_dir, exist_ok=True)
        print(f"✅ 创建数据目录: {data_dir}")
        print(f"✅ 创建预训练模型目录: {pretrained_dir}")

        # 修复配置文件路径
        config_file = self.spt_root / "experiments" / "spt" / "unimod1k.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # 更新路径
                if 'MODEL' in config:
                    if 'PRETRAINED' in config['MODEL']:
                        config['MODEL']['PRETRAINED'] = str(pretrained_dir / 'STARKS_ep0500.pth.tar')
                    if 'LANGUAGE' in config['MODEL']:
                        config['MODEL']['LANGUAGE']['PATH'] = str(pretrained_dir / 'bert-base-uncased')
                        config['MODEL']['LANGUAGE']['VOCAB_PATH'] = str(pretrained_dir / 'bert-base-uncased-vocab.txt')

                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"✅ 更新配置文件: {config_file}")
            except Exception as e:
                print(f"⚠️ 更新配置文件失败: {e}")
        else:
            print(f"⚠️ 配置文件不存在: {config_file}")

        # 运行路径设置脚本
        create_local_script = self.spt_root / "tracking" / "create_default_local_file.py"
        if create_local_script.exists():
            try:
                subprocess.run([
                    sys.executable, str(create_local_script),
                    "--workspace_dir", str(self.spt_root),
                    "--data_dir", str(data_dir),
                    "--save_dir", str(self.spt_root)
                ], check=True, cwd=str(self.spt_root))
                print("✅ 成功设置项目路径")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ 路径设置失败: {e}")
            except Exception as e:
                print(f"⚠️ 路径设置异常: {e}")
        else:
            print(f"⚠️ 路径设置脚本不存在: {create_local_script}")

    def check_structure(self):
        """检查项目结构"""
        print("🔍 检查项目结构...")

        important_paths = [
            self.spt_root / "lib",
            self.spt_root / "experiments",
            self.spt_root / "tracking",
            self.spt_root / "lib" / "models",
            self.spt_root / "lib" / "train",
            self.spt_root / "lib" / "test",
        ]

        for path in important_paths:
            if path.exists():
                print(f"✅ {path}")
            else:
                print(f"❌ {path}")

    def download_pretrained_models(self, pretrained_dir):
        """下载预训练模型的指导"""
        print("📥 预训练模型下载指导...")

        print(f"""
需要手动下载以下预训练模型到 {pretrained_dir}:

1. BERT预训练权重:
   方法1 - 手动下载:
   - URL: https://drive.google.com/drive/folders/1Fi-4TSaIP4B_TPi2Jme2sxZRdH9l5NPN?usp=share_link
   - 文件: bert-base-uncased.tar.gz, bert-base-uncased-vocab.txt

   方法2 - 使用transformers自动下载 (推荐):
   pip install transformers
   python3 -c "
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.save_pretrained('{pretrained_dir}/bert-base-uncased')
tokenizer.save_pretrained('{pretrained_dir}/bert-base-uncased')
print('BERT模型下载完成!')
"

2. STARK-S模型:
   - URL: https://drive.google.com/drive/folders/142sMjoT5wT6CuRiFT5LLejgr7VLKmaC4
   - 文件: STARKS_ep0500.pth.tar

3. 如果网络问题，可以使用国内镜像:
   - 使用清华镜像: pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
        """)

    def create_simple_test(self):
        """创建简单的测试脚本"""
        print("🧪 创建测试脚本...")

        test_code = f'''#!/usr/bin/env python3
"""
简单的环境测试脚本
"""
import sys
import os
sys.path.append('{self.spt_root}')

def test_imports():
    """测试关键模块导入"""
    print("🧪 测试模块导入...")

    try:
        import torch
        print(f"✅ PyTorch: {{torch.__version__}}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {{e}}")

    try:
        import cv2
        print(f"✅ OpenCV: {{cv2.__version__}}")
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {{e}}")

    try:
        import yaml
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML导入失败: {{e}}")

    try:
        from easydict import EasyDict
        print("✅ EasyDict")
    except ImportError as e:
        print(f"❌ EasyDict导入失败: {{e}}")

    # 测试项目模块
    try:
        from lib.config.spt.config import cfg
        print("✅ SPT配置模块")
    except ImportError as e:
        print(f"❌ SPT配置模块导入失败: {{e}}")

    try:
        from lib.models.spt import build_spt
        print("✅ SPT模型模块")
    except ImportError as e:
        print(f"❌ SPT模型模块导入失败: {{e}}")

def test_cuda():
    """测试CUDA可用性"""
    print("\\n🔥 测试CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用, 设备数量: {{torch.cuda.device_count()}}")
            print(f"✅ 当前设备: {{torch.cuda.get_device_name()}}")
        else:
            print("⚠️ CUDA不可用，将使用CPU")
    except Exception as e:
        print(f"❌ CUDA测试失败: {{e}}")

if __name__ == "__main__":
    print("🚀 开始环境测试...")
    test_imports()
    test_cuda()
    print("\\n✅ 环境测试完成!")
'''

        test_file = self.spt_root / "test_environment.py"
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)
            print(f"✅ 创建测试脚本: {test_file}")

            # 添加执行权限
            os.chmod(test_file, 0o755)

        except Exception as e:
            print(f"⚠️ 创建测试脚本失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='UniMod1K项目修复工具 v2.0')
    parser.add_argument('--project_root', default='.', help='项目根目录')
    parser.add_argument('--data_dir', help='数据目录路径')
    parser.add_argument('--pretrained_dir', help='预训练模型目录路径')
    parser.add_argument('--skip_download_info', action='store_true', help='跳过下载信息显示')

    args = parser.parse_args()

    try:
        fixer = UniMod1KFixer(args.project_root)

        print("🚀 开始修复UniMod1K项目...")

        # 检查项目结构
        fixer.check_structure()

        # 修复环境
        fixer.fix_environment()

        # 修复路径
        fixer.fix_paths(args.data_dir, args.pretrained_dir)

        # 创建测试脚本
        fixer.create_simple_test()

        # 下载指导
        if not args.skip_download_info:
            pretrained_dir = args.pretrained_dir or (Path(args.project_root) / "pretrained_models")
            fixer.download_pretrained_models(pretrained_dir)

        print("\n✅ UniMod1K项目修复完成!")
        print("\n📋 下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行环境测试: python3 test_environment.py")
        print("3. 下载预训练模型")
        print("4. 准备训练数据")

    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())