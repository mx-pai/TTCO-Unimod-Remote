#!/usr/bin/env python3
"""
SPT训练checkpoint自动清理脚本
保留最新的N个checkpoint，删除旧的以节省磁盘空间
"""

import os
import glob
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime

class CheckpointCleaner:
    def __init__(self, checkpoint_dir, keep_last=5, interval_hours=2):
        """
        Args:
            checkpoint_dir: checkpoint目录路径
            keep_last: 保留最新的N个checkpoint
            interval_hours: 清理间隔(小时)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = keep_last
        self.interval_hours = interval_hours
        self.running = False

    def get_checkpoint_files(self):
        """获取所有checkpoint文件，按epoch排序"""
        pattern = str(self.checkpoint_dir / "**/*_ep*.pth.tar")
        checkpoint_files = glob.glob(pattern, recursive=True)

        # 按文件名中的epoch编号排序
        def extract_epoch(filepath):
            filename = os.path.basename(filepath)
            try:
                # 从文件名提取epoch: SPT_ep0042.pth.tar -> 42
                epoch_part = filename.split('_ep')[1].split('.')[0]
                return int(epoch_part)
            except:
                return 0

        checkpoint_files.sort(key=extract_epoch)
        return checkpoint_files

    def clean_old_checkpoints(self):
        """清理旧的checkpoint文件"""
        checkpoint_files = self.get_checkpoint_files()

        if len(checkpoint_files) <= self.keep_last:
            print(f"📁 当前有{len(checkpoint_files)}个checkpoint，无需清理")
            return

        # 计算需要删除的文件
        files_to_delete = checkpoint_files[:-self.keep_last]
        total_size = 0

        print(f"🧹 开始清理checkpoint文件...")
        print(f"📊 总计: {len(checkpoint_files)} 个文件")
        print(f"🔄 保留: {self.keep_last} 个最新文件")
        print(f"🗑️ 删除: {len(files_to_delete)} 个旧文件")

        for file_path in files_to_delete:
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                os.remove(file_path)
                print(f"  ✅ 已删除: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
            except Exception as e:
                print(f"  ❌ 删除失败: {file_path} - {e}")

        print(f"💾 释放空间: {total_size/1024/1024:.1f}MB")
        print(f"⏰ 清理完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def start_auto_cleanup(self):
        """启动自动清理"""
        self.running = True
        print(f"🚀 启动自动清理服务")
        print(f"📁 监控目录: {self.checkpoint_dir}")
        print(f"🔄 保留数量: {self.keep_last} 个")
        print(f"⏰ 清理间隔: {self.interval_hours} 小时")

        def cleanup_loop():
            while self.running:
                try:
                    self.clean_old_checkpoints()
                    # 等待指定的时间间隔
                    time.sleep(self.interval_hours * 3600)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ 清理过程出错: {e}")
                    time.sleep(300)  # 出错后等待5分钟再试

        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def stop_auto_cleanup(self):
        """停止自动清理"""
        self.running = False
        print("🛑 停止自动清理服务")

    def manual_cleanup(self):
        """手动执行一次清理"""
        print("🔧 手动执行checkpoint清理...")
        self.clean_old_checkpoints()

def main():
    parser = argparse.ArgumentParser(description='SPT Checkpoint自动清理工具')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='./checkpoints_improved/checkpoints/train/spt/unimod1k_improved',
                       help='Checkpoint目录路径')
    parser.add_argument('--keep', type=int, default=3,
                       help='保留最新的N个checkpoint (默认: 5)')
    parser.add_argument('--interval', type=float, default=1,
                       help='清理间隔(小时) (默认: 2.0)')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                       help='运行模式: auto(自动) 或 manual(手动)')

    args = parser.parse_args()

    cleaner = CheckpointCleaner(
        checkpoint_dir=args.checkpoint_dir,
        keep_last=args.keep,
        interval_hours=args.interval
    )

    if args.mode == 'manual':
        cleaner.manual_cleanup()
    else:
        cleaner.start_auto_cleanup()
        try:
            print("按 Ctrl+C 停止自动清理...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            cleaner.stop_auto_cleanup()
            print("\n👋 程序已退出")

if __name__ == "__main__":
    main()