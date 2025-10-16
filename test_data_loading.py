#!/usr/bin/env python3
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
