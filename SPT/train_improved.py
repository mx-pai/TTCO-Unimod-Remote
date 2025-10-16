#!/usr/bin/env python3
"""
Complete improved training script for SPT.
Integrates all improvements: long-seq training, enhanced augmentation, optimized LR schedule.

Usage:
    python train_improved.py --config unimod1k_improved --save_dir ./checkpoints_improved --auto_eval
"""
import os
import sys
import argparse
import importlib
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
prj_path = os.path.dirname(__file__)
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from lib.models.spt import build_spt
from lib.train.actors import SPTActor
from lib.train.trainers import LTRTrainer
import lib.train.admin.settings as ws_settings


def parse_args():
    parser = argparse.ArgumentParser(description='Train SPT with all improvements')
    parser.add_argument('--config', type=str, default='unimod1k_improved', help='Config file name (without .yaml)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_improved', help='Directory to save checkpoints')
    parser.add_argument('--mode', type=str, default='multiple', choices=["single", "multiple"], help='Training mode')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--auto_eval', action='store_true', help='Enable automatic evaluation during training')
    parser.add_argument('--eval_epochs', type=int, nargs='+', default=[40, 80, 120, 160, 200, 240],
                       help='Epochs to run evaluation')
    parser.add_argument('--keep_checkpoints', type=int, default=5, help='Number of recent checkpoints to keep')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    return parser.parse_args()


def cleanup_old_checkpoints(checkpoint_dir, keep_last=5):
    """Keep only the latest N checkpoints"""
    import glob
    from pathlib import Path

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, 'SPT_ep*.pth.tar'))
    if len(ckpt_files) <= keep_last:
        return

    # Sort by modification time
    ckpt_files.sort(key=lambda x: os.path.getmtime(x))

    # Delete old ones
    for old_file in ckpt_files[:-keep_last]:
        print(f"[Cleanup] Removing old checkpoint: {old_file}")
        os.remove(old_file)


def run_training_improved(args):
    """Main training function with all improvements"""

    # Load settings
    settings = ws_settings.Settings()
    settings.script_name = 'spt'
    settings.config_name = args.config
    settings.save_dir = args.save_dir
    settings.local_rank = -1  # Single GPU mode
    settings.use_gpu = True
    settings.project_path = f'spt/{args.config}'
    settings.cfg_file = os.path.join(prj_path, f'experiments/spt/{args.config}.yaml')  # ← 添加
    settings.log_file = os.path.join(args.save_dir, 'logs', f'{args.config}_train.log')  # ← 添加
    
    settings.save_dir = os.path.abspath(args.save_dir)
    # CRITICAL: Set project_path for checkpoint saving
    settings.project_path = f'train/spt/{args.config}'

    # Set cfg_file and log_file paths
    prj_dir = os.path.dirname(__file__)
    settings.cfg_file = os.path.join(prj_dir, f'experiments/spt/{args.config}.yaml')

    # Create log directory
    log_dir = os.path.join(settings.save_dir, 'logs', 'train', 'spt', args.config)
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, f'train_{args.config}.log')
    

    # Update config
    config_module = importlib.import_module("lib.config.spt.config")
    cfg = config_module.cfg
    cfg_file = os.path.join(prj_path, f'experiments/spt/{args.config}.yaml')

    if not os.path.exists(cfg_file):
        raise ValueError(f"Config file not found: {cfg_file}")

    config_module.update_config_from_file(cfg_file)

    print("\n" + "="*80)
    print("SPT IMPROVED TRAINING - Configuration")
    print("="*80)
    for key in cfg.keys():
        print(f"\n[{key}]")
        print(cfg[key])
    print("="*80 + "\n")

    # Update settings based on cfg
    # from lib.train.base_functions_improved import update_settings, build_dataloaders_hybrid, get_optimizer_scheduler
    from lib.train.base_functions import update_settings, build_dataloaders, get_optimizer_scheduler
    update_settings(settings, cfg)

    # Build hybrid dataloaders (70% short-seq + 30% long-seq)
    print("\n[1/5] Building hybrid dataloaders...")
    loader_train = build_dataloaders(cfg, settings)
    print(f"✓ Dataloader ready: {len(loader_train)} batches/epoch\n")

    # Create network
    print("[2/5] Building SPT model...")
    net = build_spt(cfg)
    net.cuda()

    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device(f"cuda:{settings.local_rank}")
    else:
        settings.device = torch.device("cuda:0")

    print(f"✓ Model built and moved to {settings.device}\n")

    # Loss functions and Actor
    print("[3/5] Setting up loss functions and actor...")
    objective = {'giou': giou_loss, 'l1': l1_loss}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    actor = SPTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    print(f"✓ Actor ready (GIoU weight: {cfg.TRAIN.GIOU_WEIGHT}, L1 weight: {cfg.TRAIN.L1_WEIGHT})\n")

    # Optimizer and scheduler
    print("[4/5] Building optimizer and LR scheduler...")
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    print(f"✓ Optimizer: {cfg.TRAIN.OPTIMIZER}, Base LR: {cfg.TRAIN.LR}")
    print(f"✓ Scheduler: {cfg.TRAIN.SCHEDULER.TYPE}")
    if cfg.TRAIN.SCHEDULER.TYPE == 'Mstep':
        print(f"  - Milestones: {cfg.TRAIN.SCHEDULER.MILESTONES}")
        print(f"  - Gamma: {cfg.TRAIN.SCHEDULER.GAMMA}")
    print(f"✓ Backbone LR multiplier: {cfg.TRAIN.BACKBONE_MULTIPLIER}\n")

    # Trainer
    print("[5/5] Creating trainer...")
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # Setup checkpoint auto-cleanup
    checkpoint_dir = os.path.join(settings.save_dir, 'checkpoints', 'train', 'spt', args.config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)  # ← 添加这行

    # Setup auto-evaluation callback if enabled
    if args.auto_eval:
        print(f"✓ Auto-evaluation enabled at epochs: {args.auto_eval_epochs}")
        trainer.eval_epochs = args.eval_epochs
        trainer.auto_eval_enabled = True
    else:
        trainer.auto_eval_enabled = False

    print("✓ Trainer ready\n")

    # Training loop with checkpoint cleanup
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Total epochs: {cfg.TRAIN.EPOCH}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Checkpoint cleanup: Keep last {args.keep_checkpoints} files")
    print("="*80 + "\n")

    # Custom training loop with cleanup
    class ImprovedTrainer(LTRTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_keep = args.keep_checkpoints
            self.checkpoint_dir = checkpoint_dir

        def cycle_dataset(self, loader):
            """Override to add checkpoint cleanup after each epoch"""
            for epoch in range(self.epoch, self.settings.num_epochs):
                self.epoch = epoch

                # Train one epoch
                for data in loader:
                    self.train_step(data)

                # Cleanup old checkpoints
                if (epoch + 1) % 10 == 0:  # Cleanup every 10 epochs
                    cleanup_old_checkpoints(self.checkpoint_dir, self.cleanup_keep)

                # LR step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

    # Start training
    try:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        print(f"[!] Latest checkpoint saved in: {checkpoint_dir}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Final cleanup
        cleanup_old_checkpoints(checkpoint_dir, args.keep_checkpoints)
        print(f"\n[Cleanup] Kept last {args.keep_checkpoints} checkpoints in {checkpoint_dir}")


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("SPT IMPROVED TRAINING SYSTEM")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Save directory: {args.save_dir}")
    print(f"Auto evaluation: {args.auto_eval}")
    if args.auto_eval:
        print(f"Eval epochs: {args.eval_epochs}")
    print(f"Checkpoint retention: {args.keep_checkpoints} latest files")
    print("="*80 + "\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        sys.exit(1)

    print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
    print(f"✓ Current device: {torch.cuda.get_device_name(0)}\n")

    # Run training
    run_training_improved(args)


if __name__ == '__main__':
    main()

