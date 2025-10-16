"""
Improved training functions with hybrid sampling (short-seq + long-seq).
"""
import torch
import random
from torch.utils.data.distributed import DistributedSampler
# from lib.train.dataset import sampler, VLTrackingSampler, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
from lib.train.data import LTRLoader  
from lib.train.data.sampler import VLTrackingSampler  

def build_dataloaders_hybrid(cfg, settings):
    """
    Build hybrid dataloaders: 70% short-seq (2 frames) + 30% long-seq (3-5 frames)
    """
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),  # Color jitter
        tfm.RandomHorizontalFlip(probability=0.5)
    )

    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.2))

    # Settings for data sampling
    settings.num_template = getattr(cfg.DATA.TEMPLATE, 'NUMBER', 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, 'NUMBER', 1)
    settings.max_gap = cfg.DATA.MAX_SAMPLE_INTERVAL
    settings.max_seq_len = getattr(cfg.DATA, 'MAX_SEQ_LENGTH', 40)

    # Import dataset modules
    from lib.train import dataset as datasets_module

    # Build short-seq datasets
    short_seq_datasets = []
    for dataset_name in cfg.DATA.TRAIN.DATASETS_NAME:
        dataset_cls = getattr(datasets_module, dataset_name, None)
        if dataset_cls is None:
            raise ValueError(f"Dataset {dataset_name} not found")

        # Short-seq dataset (original 2-frame sampling)
        short_dataset = dataset_cls(
            root=settings.env.unimod1k_dir if dataset_name == 'UniMod1K' else None,
            dtype='rgbcolormap'
        )
        short_seq_datasets.append(short_dataset)

    # Build long-seq datasets (same datasets, different sampler)
    long_seq_datasets = short_seq_datasets.copy()

    # Build processing
    from lib.train.data import processing
    data_processing_train = processing.SPTProcessing(
        search_area_factor=cfg.DATA.SEARCH.FACTOR,
        template_area_factor=cfg.DATA.TEMPLATE.FACTOR,
        search_sz=cfg.DATA.SEARCH.SIZE,
        temp_sz=cfg.DATA.TEMPLATE.SIZE,
        center_jitter_factor={'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                              'search': cfg.DATA.SEARCH.CENTER_JITTER},
        scale_jitter_factor={'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                             'search': cfg.DATA.SEARCH.SCALE_JITTER},
        mode='sequence',
        transform=transform_train,
        joint_transform=transform_joint,
        settings=settings
    )

    # Short-seq sampler (original VLTrackingSampler)
    # from lib.train.data.sampler import VLTrackingSampler

    short_samples_per_epoch = int(cfg.DATA.TRAIN.SAMPLE_PER_EPOCH * 0.7)  # 70%
    sampler_train_short = VLTrackingSampler(
        datasets=short_seq_datasets,
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=short_samples_per_epoch,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode='causal',
        bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH
    )

    # Long-seq sampler (new anti-drift sampler)
    from lib.train.data.sampler_longseq import LongSeqTrackingSampler

    long_samples_per_epoch = int(cfg.DATA.TRAIN.SAMPLE_PER_EPOCH * 0.3)  # 30%
    long_seq_length = getattr(cfg.DATA.TRAIN, 'LONG_SEQ_LENGTH', 4)

    sampler_train_long = LongSeqTrackingSampler(
        datasets=long_seq_datasets,
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=long_samples_per_epoch,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=1,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        seq_length=long_seq_length,
        bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH
    )

    # Combine samplers using ConcatDataset
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([sampler_train_short, sampler_train_long])

    # Build dataloader
    train_sampler = DistributedSampler(combined_dataset) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader(
        'train',
        combined_dataset,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=train_sampler,
        epoch_interval=1
    )

    if is_main_process():
        print(f"[DataLoader] Short-seq samples: {short_samples_per_epoch}, Long-seq samples: {long_samples_per_epoch}")
        print(f"[DataLoader] Total batches per epoch: {len(loader_train)}")

    return loader_train


def update_settings(settings, cfg):
    """Update settings based on config"""
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def get_optimizer_scheduler(net, cfg):
    """Build optimizer and scheduler with separate LR for different components"""
    # Separate parameters into groups
    param_dicts = [
        {
            "params": [p for n, p in net.named_parameters()
                      if "backbone" not in n and "language_backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in net.named_parameters()
                      if "backbone" in n and "language_backbone" not in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
        {
            "params": [p for n, p in net.named_parameters()
                      if "language_backbone" in n and p.requires_grad],
            "lr": cfg.MODEL.LANGUAGE.BERT.LR if hasattr(cfg.MODEL.LANGUAGE.BERT, 'LR') else cfg.TRAIN.LR,
        }
    ]

    # Build optimizer
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.TRAIN.OPTIMIZER}")

    # Build scheduler
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.TRAIN.LR_DROP_EPOCH,
            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == 'Mstep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.TRAIN.SCHEDULER.TYPE}")

    return optimizer, lr_scheduler


def names2datasets(name_list, settings, image_loader):
    """Convert dataset names to dataset objects"""
    from lib.train import dataset as datasets_module

    datasets = []
    for name in name_list:
        dataset_cls = getattr(datasets_module, name, None)
        if dataset_cls is None:
            raise ValueError(f"Dataset {name} not found")

        if name == 'UniMod1K':
            dataset = dataset_cls(
                root=settings.env.unimod1k_dir,
                dtype='rgbcolormap',
                image_loader=image_loader
            )
        else:
            dataset = dataset_cls(image_loader=image_loader)

        datasets.append(dataset)

    return datasets

