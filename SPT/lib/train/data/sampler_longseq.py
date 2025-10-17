import random
from typing import List, Optional

import torch
import torch.utils.data

from lib.train.data.sampler import VLTrackingSampler
from lib.utils import TensorDict


class LongSeqTrackingSampler(VLTrackingSampler):
    """Sampler that generates long temporal snippets (template + consecutive search frames).

    The implementation reuses the utilities from :class:`VLTrackingSampler` but enforces
    consecutive search frames of length ``seq_length`` to encourage long-term supervision.
    """

    def __init__(self, *args,
                 seq_length: int = 4,
                 **kwargs):
        if seq_length < 1:
            raise ValueError("seq_length must be >= 1")

        # Force positive sampling and set number of search frames to the sequence length.
        kwargs.setdefault('pos_prob', 1.0)
        kwargs['num_search_frames'] = seq_length

        super().__init__(*args, **kwargs)
        self.seq_length = seq_length

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        if self.train_cls:
            # Classification mode not supported for long sequence sampling at the moment.
            raise RuntimeError("LongSeqTrackingSampler does not support classification sampling.")
        return self._sample_long_sequence()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _sample_long_sequence(self) -> TensorDict:
        valid = False
        data = None

        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            if not dataset.is_video_sequence():
                # Fallback to the parent sampler for image datasets.
                data = super().getitem(0)
                valid = True
                break

            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, True)
            candidate = self._find_candidate_index(visible)
            if candidate is None:
                continue

            template_ids = self._build_template_ids(candidate, visible)
            if template_ids is None:
                continue

            search_ids = list(range(candidate + 1, candidate + 1 + self.seq_length))

            try:
                template_frames, template_anno, _ = dataset.get_frames(seq_id, template_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_ids, seq_info_dict)

                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [
                    torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [
                    torch.zeros((H, W))] * self.num_search_frames

                nl = template_anno.get('nlp', [None])[0]
                tracking_nl_token_ids, tracking_nl_token_masks = self.extract_token_from_nlp(
                    nl if nl is not None else "", self.max_seq_len)

                data = TensorDict({
                    'template_images': template_frames,
                    'template_anno': template_anno['bbox'],
                    'template_masks': template_masks,
                    'search_images': search_frames,
                    'search_anno': search_anno['bbox'],
                    'search_masks': search_masks,
                    'nl_token_ids': tracking_nl_token_ids,
                    'nl_token_masks': tracking_nl_token_masks,
                    'dataset': dataset.get_name(),
                    'test_class': meta_obj_test.get('object_class_name')
                })

                data = self.processing(data)
                valid = data['valid']
            except Exception:
                valid = False

        return data

    def _find_candidate_index(self, visible: torch.Tensor) -> Optional[int]:
        """Find a starting index that supports seq_length consecutive visible frames."""
        if visible.ndim != 1:
            visible = visible.flatten()
        vis = visible.bool()

        max_start = vis.numel() - (self.seq_length + 1)
        if max_start <= 0:
            return None

        candidates: List[int] = []
        for idx in range(max_start):
            window = vis[idx: idx + self.seq_length + 1]
            if bool(torch.all(window)):
                candidates.append(idx)

        if not candidates:
            return None
        return random.choice(candidates)

    def _build_template_ids(self, base_idx: int, visible: torch.Tensor) -> Optional[List[int]]:
        """Compose template frame ids, ensuring visibility."""
        template_ids = [base_idx]
        if self.num_template_frames == 1:
            return template_ids

        extra = self._sample_visible_ids(
            visible,
            num_ids=self.num_template_frames - 1,
            min_id=max(0, base_idx - self.max_gap),
            max_id=base_idx,
            allow_invisible=False
        )

        if extra is None:
            return None

        template_ids = sorted(extra + [base_idx])
        return template_ids
