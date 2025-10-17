#!/usr/bin/env python3
"""
Unified training entrypoint for the SPT tracker.

This script centralises data-loader construction (including optional long-sequence
sampling), model instantiation, optimisation setup, and checkpoint housekeeping.
"""

import argparse
import glob
import importlib
import os
import sys
from typing import Tuple

import torch
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------------------------------------------------------- #
# Path setup
# ----------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------------- #
# Local imports (after sys.path adjustment)
# ----------------------------------------------------------------------------- #
from lib.models.spt import build_spt
from lib.train.actors import SPTActor
from lib.train.trainers import LTRTrainer
import lib.train.admin.settings as ws_settings
from lib.utils.box_ops import giou_loss
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler, update_settings


# ----------------------------------------------------------------------------- #
# Argument parsing
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SPT tracker.")
    parser.add_argument("--config", default="unimod1k_improved",
                        help="Config file name (without extension) located under --config-dir.")
    parser.add_argument("--config-dir", default="experiments/spt",
                        help="Directory (relative to project root) containing tracker configs.")
    parser.add_argument("--save-dir", default="./outputs",
                        help="Base output directory for checkpoints, logs, and tensorboard files.")
    parser.add_argument("--data-root", default=None,
                        help="Path to the UniMod1K dataset root. Overrides env_settings().unimod1k_dir.")
    parser.add_argument("--nlp-root", default=None,
                        help="Path to the UniMod1K NLP annotations root. Defaults to --data-root if omitted.")
    parser.add_argument("--resume", default=None,
                        help="Optional checkpoint path to resume from. If omitted, the latest checkpoint "
                             "in the project path is used when available.")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                        help="Number of most recent checkpoints to keep after each training run.")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the merged training configuration before starting.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed-precision training even if the config enables it.")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="(Optional) Local rank for distributed launch. -1 means single GPU.")
    return parser.parse_args()


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def load_config(config_name: str, config_dir: str) -> Tuple[object, str]:
    """Load and return the runtime config object and its absolute path."""
    cfg_module = importlib.import_module("lib.config.spt.config")
    cfg = cfg_module.cfg

    cfg_path = os.path.join(PROJECT_ROOT, config_dir, f"{config_name}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg_module.update_config_from_file(cfg_path)
    return cfg, cfg_path


def init_settings(args: argparse.Namespace, cfg_path: str) -> ws_settings.Settings:
    """Initialise shared training settings."""
    settings = ws_settings.Settings()
    settings.script_name = "spt"
    settings.config_name = args.config
    settings.cfg_file = cfg_path
    settings.save_dir = os.path.abspath(args.save_dir)
    settings.project_path = os.path.join("train", "spt", args.config)
    settings.local_rank = args.local_rank
    settings.use_gpu = True
    if args.data_root:
        settings.data_root = os.path.abspath(args.data_root)
    if args.nlp_root:
        settings.nlp_root = os.path.abspath(args.nlp_root)
    elif args.data_root:
        settings.nlp_root = os.path.abspath(args.data_root)

    log_dir = os.path.join(settings.save_dir, "logs", settings.project_path)
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, "train.log")
    return settings


def ensure_checkpoint_dir(settings: ws_settings.Settings) -> str:
    ckpt_dir = os.path.join(settings.save_dir, "checkpoints", settings.project_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """Remove old checkpoints, keeping only the most recent `keep_last` files."""
    if keep_last <= 0:
        return

    pattern = os.path.join(checkpoint_dir, "SPT_ep*.pth.tar")
    checkpoints = sorted([p for p in glob.glob(pattern)], key=os.path.getmtime)
    if len(checkpoints) <= keep_last:
        return

    for path in checkpoints[:-keep_last]:
        try:
            os.remove(path)
            print(f"[cleanup] removed old checkpoint: {path}")
        except OSError as exc:
            print(f"[cleanup] failed to remove {path}: {exc}")


def maybe_wrap_ddp(net: torch.nn.Module, settings: ws_settings.Settings) -> torch.nn.Module:
    if settings.local_rank == -1:
        settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return net.to(settings.device)

    torch.cuda.set_device(settings.local_rank)
    settings.device = torch.device(f"cuda:{settings.local_rank}")
    net = net.to(settings.device)
    return DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)


def print_config(cfg: object) -> None:
    print("\n" + "=" * 80)
    print("Merged configuration")
    print("=" * 80)
    for key in cfg.keys():
        print(f"\n[{key}]")
        print(cfg[key])
    print("=" * 80 + "\n")


# ----------------------------------------------------------------------------- #
# Main routine
# ----------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but was not detected.")

    print(f"✓ CUDA devices: {torch.cuda.device_count()}")
    print(f"✓ Primary GPU: {torch.cuda.get_device_name(0)}")

    cfg, cfg_path = load_config(args.config, args.config_dir)
    if args.print_config:
        print_config(cfg)

    settings = init_settings(args, cfg_path)
    update_settings(settings, cfg)

    if getattr(settings, 'data_root', None):
        print(f"✓ Data root: {settings.data_root}")
    elif getattr(settings.env, 'unimod1k_dir', None):
        print(f"✓ Data root: {settings.env.unimod1k_dir} (from env settings)")
    else:
        print("[!] Data root not set, please provide --data-root or update env settings.")

    # Build dataloaders
    loader_train = build_dataloaders(cfg, settings)
    print(f"✓ Training batches per epoch: {len(loader_train)}")

    # Build model & actor
    net = build_spt(cfg)
    net = maybe_wrap_ddp(net, settings)

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    objective = {'giou': giou_loss, 'l1': l1_loss}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
    actor = SPTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = bool(getattr(cfg.TRAIN, "AMP", False) and not args.no_amp)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    checkpoint_dir = ensure_checkpoint_dir(settings)
    resume_latest = False
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        resume_latest = True

    try:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=resume_latest, fail_safe=True)
    finally:
        cleanup_old_checkpoints(checkpoint_dir, args.keep_checkpoints)


if __name__ == "__main__":
    main()
