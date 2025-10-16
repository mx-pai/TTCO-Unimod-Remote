#!/usr/bin/env python3
"""
比赛数据导入和格式转换指南
将比赛数据格式转换为UniMod1K/SPT可以使用的格式
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import json

class CompetitionDataProcessor:
    """比赛数据处理器"""

    def __init__(self, competition_data_root, output_root):
        """
        初始化数据处理器
        Args:
            competition_data_root: 比赛数据根目录
            output_root: 输出数据根目录（UniMod1K格式）
        """
        self.competition_root = Path(competition_data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        print(f"📂 比赛数据目录: {self.competition_root}")
        print(f"📂 输出数据目录: {self.output_root}")

    def analyze_competition_structure(self):
        """分析比赛数据结构"""
        print("🔍 分析比赛数据结构...")

        if not self.competition_root.exists():
            print(f"❌ 比赛数据目录不存在: {self.competition_root}")
            return False

        # 列出所有文件和目录
        print("\n📁 数据目录结构:")
        for item in self.competition_root.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(self.competition_root)
                print(f"📄 {rel_path}")
            elif item.is_dir():
                rel_path = item.relative_to(self.competition_root)
                print(f"📁 {rel_path}/")

        return True

    def convert_sequence(self, seq_name, rgb_dir, depth_dir, text_file, gt_file):
        """
        转换单个序列到UniMod1K格式
        Args:
            seq_name: 序列名称
            rgb_dir: RGB图像目录
            depth_dir: 深度图像目录
            text_file: 文本描述文件
            gt_file: 标注文件
        """
        print(f"🔄 转换序列: {seq_name}")

        # 创建输出目录
        seq_output = self.output_root / seq_name
        seq_output.mkdir(parents=True, exist_ok=True)

        color_output = seq_output / "color"
        depth_output = seq_output / "depth"
        color_output.mkdir(exist_ok=True)
        depth_output.mkdir(exist_ok=True)

        # 转换RGB图像
        rgb_files = self._get_image_files(rgb_dir, ['.jpg', '.jpeg', '.png'])
        print(f"📸 RGB图像数量: {len(rgb_files)}")

        for i, rgb_file in enumerate(rgb_files):
            dst_name = f"{i+1:08d}.jpg"
            dst_path = color_output / dst_name
            if not dst_path.exists():
                shutil.copy2(rgb_file, dst_path)

        # 转换深度图像
        depth_files = self._get_image_files(depth_dir, ['.png'])
        print(f"🌊 深度图像数量: {len(depth_files)}")

        for i, depth_file in enumerate(depth_files):
            dst_name = f"{i+1:08d}.png"
            dst_path = depth_output / dst_name
            if not dst_path.exists():
                shutil.copy2(depth_file, dst_path)

        # 复制文本描述
        if Path(text_file).exists():
            shutil.copy2(text_file, seq_output / "nlp.txt")
            print("✅ 文本描述已复制")

        # 复制标注文件
        if Path(gt_file).exists():
            shutil.copy2(gt_file, seq_output / "groundtruth_rect.txt")
            print("✅ 标注文件已复制")

        print(f"✅ 序列 {seq_name} 转换完成")
        return True

    def _get_image_files(self, directory, extensions):
        """获取指定扩展名的图像文件"""
        directory = Path(directory)
        if not directory.exists():
            return []

        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(files)

    def batch_convert_from_structure(self, train_or_test="train"):
        """
        根据比赛数据结构批量转换
        假设比赛数据结构为:
        competition_data/
        ├── train/ (或 test/)
        │   ├── seq001/
        │   │   ├── rgb/
        │   │   ├── depth/
        │   │   ├── text.txt
        │   │   └── gt.txt
        │   └── seq002/
        """
        print(f"🔄 批量转换{train_or_test}数据...")

        data_dir = self.competition_root / train_or_test
        if not data_dir.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            return False

        # 遍历所有序列目录
        seq_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"📊 找到 {len(seq_dirs)} 个序列")

        for seq_dir in seq_dirs:
            seq_name = seq_dir.name

            # 查找RGB和深度目录
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            text_file = seq_dir / "text.txt"
            gt_file = seq_dir / "gt.txt"

            # 检查必需文件是否存在
            if not rgb_dir.exists():
                print(f"⚠️ RGB目录不存在: {seq_name}")
                continue

            if not depth_dir.exists():
                print(f"⚠️ 深度目录不存在: {seq_name}")
                continue

            # 转换序列
            self.convert_sequence(seq_name, rgb_dir, depth_dir, text_file, gt_file)

        print(f"✅ {train_or_test}数据批量转换完成")
        return True

    def create_dataset_split(self):
        """创建数据集划分文件"""
        print("📝 创建数据集划分文件...")

        sequences = []
        for seq_dir in self.output_root.iterdir():
            if seq_dir.is_dir():
                sequences.append(seq_dir.name)

        sequences.sort()

        # 创建训练集划分（假设前80%为训练，后20%为验证）
        split_point = int(len(sequences) * 0.8)
        train_seqs = sequences[:split_point]
        val_seqs = sequences[split_point:]

        # 保存划分文件
        split_info = {
            'train': train_seqs,
            'val': val_seqs,
            'total': len(sequences)
        }

        with open(self.output_root / "dataset_split.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"✅ 数据集划分: 训练{len(train_seqs)}, 验证{len(val_seqs)}")
        return split_info

    def verify_converted_data(self):
        """验证转换后的数据"""
        print("🔍 验证转换后的数据...")

        sequences = [d for d in self.output_root.iterdir() if d.is_dir()]
        print(f"📊 总序列数: {len(sequences)}")

        for seq_dir in sequences[:3]:  # 验证前3个序列
            print(f"\n🔍 验证序列: {seq_dir.name}")

            # 检查目录结构
            color_dir = seq_dir / "color"
            depth_dir = seq_dir / "depth"
            nlp_file = seq_dir / "nlp.txt"
            gt_file = seq_dir / "groundtruth_rect.txt"

            if color_dir.exists():
                color_files = len(list(color_dir.glob("*.jpg")))
                print(f"  ✅ RGB图像: {color_files} 张")
            else:
                print(f"  ❌ RGB目录不存在")

            if depth_dir.exists():
                depth_files = len(list(depth_dir.glob("*.png")))
                print(f"  ✅ 深度图像: {depth_files} 张")
            else:
                print(f"  ❌ 深度目录不存在")

            if nlp_file.exists():
                with open(nlp_file, 'r') as f:
                    text = f.read().strip()
                print(f"  ✅ 文本描述: {text[:50]}...")
            else:
                print(f"  ❌ 文本描述不存在")

            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    lines = f.readlines()
                print(f"  ✅ 标注文件: {len(lines)} 帧")
            else:
                print(f"  ❌ 标注文件不存在")

        print("✅ 数据验证完成")

def create_data_loader_test():
    """创建数据加载测试脚本"""

    test_code = '''#!/usr/bin/env python3
"""
测试数据加载
"""
import sys
import os
from pathlib import Path

# 添加SPT路径
spt_path = "/root/autodl-tmp/UniMod1K/SPT"
sys.path.insert(0, spt_path)
sys.path.insert(0, os.path.join(spt_path, "lib"))

def test_data_loading():
    """测试数据加载"""
    print("🧪 测试数据加载...")

    try:
        from lib.train.dataset.unimod1k import UniMod1K
        print("✅ UniMod1K数据集类导入成功")

        # 测试数据集初始化
        data_root = "/root/autodl-tmp/competition_data_converted"
        if Path(data_root).exists():
            dataset = UniMod1K(root=data_root, nlp_root=data_root)
            print(f"✅ 数据集初始化成功，序列数: {len(dataset.sequence_list)}")

            # 测试加载第一个样本
            if len(dataset.sequence_list) > 0:
                seq_name = dataset.sequence_list[0]
                print(f"✅ 测试序列: {seq_name}")

                # 这里可以添加更多的数据加载测试
                return True
        else:
            print(f"⚠️ 数据目录不存在: {data_root}")
            return False

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
'''

    with open("test_data_loading.py", 'w') as f:
        f.write(test_code)
    print("✅ 创建数据加载测试脚本: test_data_loading.py")

def main():
    """主函数 - 数据导入指南"""
    print("🚀 比赛数据导入指南")
    print("="*50)

    print("""
📋 数据导入步骤:

1. 📥 下载比赛数据
   - 训练集: 1000个序列，60,000个RGB-Depth图像对
   - 验证集: 50个序列，11,800个图像对
   - 测试集: 50个序列

2. 📁 准备数据目录结构
   比赛数据格式:
   competition_data/
   ├── train/
   │   ├── seq001/
   │   │   ├── rgb/          # RGB图像序列
   │   │   ├── depth/        # 深度图像序列
   │   │   ├── text.txt      # 文本描述
   │   │   └── gt.txt        # 标注文件
   │   └── ...
   └── test/

3. 🔄 转换数据格式
   转换为UniMod1K格式:
   converted_data/
   ├── seq001/
   │   ├── color/            # RGB图像 (00000001.jpg, ...)
   │   ├── depth/            # 深度图像 (00000001.png, ...)
   │   ├── nlp.txt           # 文本描述
   │   └── groundtruth_rect.txt  # 标注 [x,y,w,h]
   └── ...

4. ⚙️ 配置SPT数据路径
5. 🧪 测试数据加载
    """)

    print("\n📋 使用方法:")
    print("1. 修改下面的路径为您的实际路径")
    print("2. 运行数据转换")
    print("3. 验证转换结果")

    # 示例使用
    competition_data_root = "/root/autodl-tmp/competition_data"  # 修改为您的比赛数据路径
    output_root = "/root/autodl-tmp/competition_data_converted"   # 输出路径

    print(f"\n📂 示例路径配置:")
    print(f"   比赛数据: {competition_data_root}")
    print(f"   转换输出: {output_root}")

    # 创建数据加载测试脚本
    create_data_loader_test()

    print(f"\n🔧 转换数据的Python代码:")
    print(f"""
# 创建数据处理器
processor = CompetitionDataProcessor(
    competition_data_root="{competition_data_root}",
    output_root="{output_root}"
)

# 分析数据结构
processor.analyze_competition_structure()

# 批量转换训练数据
processor.batch_convert_from_structure("train")

# 创建数据集划分
processor.create_dataset_split()

# 验证转换结果
processor.verify_converted_data()
    """)

if __name__ == "__main__":
    main()