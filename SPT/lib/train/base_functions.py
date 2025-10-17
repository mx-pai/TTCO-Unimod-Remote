import os
import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import UniMod1K
from torch.utils.data import ConcatDataset
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
from lib.train.data.sampler_longseq import LongSeqTrackingSampler


def update_settings(settings, cfg):
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


def names2datasets(name_list: list, settings, image_loader, split_files=None):
    assert isinstance(name_list, list)
    split_files = split_files or {}
    datasets = []
    for name in name_list:
        assert name in ["UniMod1K"]
        if name == 'UniMod1K':
            data_root = getattr(settings, 'data_root', None) or getattr(settings.env, 'unimod1k_dir', None)
            nlp_root = getattr(settings, 'nlp_root', None) or getattr(settings.env, 'unimod1k_dir_nlp', data_root)
            split_file = split_files.get(name)
            datasets.append(UniMod1K(root=data_root, nlp_root=nlp_root,
                                     dtype='rgbcolormap', image_loader=image_loader,
                                     split_file=split_file))

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))


    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SPTProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)


    # Train sampler and loader
    base_num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    base_num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    long_seq_ratio = max(0.0, float(getattr(cfg.DATA.TRAIN, "LONG_SEQ_RATIO", 0.0)))
    long_seq_length = max(1, int(getattr(cfg.DATA.TRAIN, "LONG_SEQ_LENGTH", 1)))
    use_long_sequence = long_seq_ratio > 0.0 and long_seq_length > 1

    if use_long_sequence:
        base_num_search = max(base_num_search, long_seq_length)

    settings.num_template = base_num_template
    settings.num_search = base_num_search

    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)

    total_samples = int(cfg.DATA.TRAIN.SAMPLE_PER_EPOCH)
    long_samples = int(total_samples * long_seq_ratio) if use_long_sequence else 0
    long_samples = min(long_samples, total_samples)
    short_samples = total_samples - long_samples

    dataset_components = []
    split_setting = getattr(cfg.DATA.TRAIN, "SPLIT_FILE", None)
    split_map = {}
    if split_setting:
        if isinstance(split_setting, (list, tuple)):
            for name, path in zip(cfg.DATA.TRAIN.DATASETS_NAME, split_setting):
                split_map[name] = path
        else:
            split_map = {cfg.DATA.TRAIN.DATASETS_NAME[0]: split_setting}

    dataset_names = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader, split_map)
    if short_samples > 0:
        dataset_components.append(
            sampler.VLTrackingSampler(
                datasets=dataset_names,
                p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                samples_per_epoch=short_samples,
                max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                num_search_frames=settings.num_search,
                num_template_frames=settings.num_template,
                processing=data_processing_train,
                frame_sample_mode=sampler_mode,
                train_cls=train_cls,
                max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
                bert_model=cfg.MODEL.LANGUAGE.TYPE,
                bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH
            )
        )

    if use_long_sequence and long_samples > 0:
        dataset_components.append(
            LongSeqTrackingSampler(
                datasets=dataset_names,
                p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                samples_per_epoch=long_samples,
                max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                num_search_frames=settings.num_search,
                num_template_frames=settings.num_template,
                processing=data_processing_train,
                frame_sample_mode=sampler_mode,
                train_cls=False,
                max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
                bert_model=cfg.MODEL.LANGUAGE.TYPE,
                bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH,
                seq_length=settings.num_search
            )
        )

    if not dataset_components:
        raise RuntimeError("No dataset components were created for training. Check SAMPLE_PER_EPOCH and ratios.")

    if len(dataset_components) == 1:
        dataset_train = dataset_components[0]
    else:
        dataset_train = ConcatDataset(dataset_components)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):

    VISUAL_LR = getattr(cfg.TRAIN, "LR", 10e-5)
    LANGUAGE_LR = getattr(cfg.MODEL.LANGUAGE.BERT, "LR", 10e-5)

    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and 'nl_pos_embed' not in n
                        and 'text_proj' not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and "language_backbone" not in n
                           and p.requires_grad],
                "lr": VISUAL_LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if ("language_backbone" in n or 'nl_pos_embed' in n
                                                                    or 'text_proj' in n) and p.requires_grad],
                "lr": LANGUAGE_LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            }
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
