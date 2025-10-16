from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class SPTLongSeqActor(BaseActor):
    """
    Actor for training SPT with long sequences (anti-drift training).

    Training strategy:
    - Template: t
    - Search: t+1, t+2, ..., t+k (consecutive frames)
    - Loss: accumulated from all search frames (simulate real tracking drift)
    """
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data with:
                - template_images: (batch, 6, 128, 128)
                - search_images_seq: List of (batch, 6, 320, 320), length=seq_len
                - search_anno_seq: List of (batch, 4), length=seq_len
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass with sequential tracking
        out_dicts, gt_bboxes_seq = self.forward_pass_longseq(data)

        # compute losses (accumulated over sequence)
        loss, status = self.compute_losses_longseq(out_dicts, gt_bboxes_seq)

        return loss, status

    def forward_pass_longseq(self, data):
        """
        Forward pass with long sequence:
        1. Extract template features once
        2. For each search frame, forward and accumulate predictions
        """
        # Process NLP
        data['nl_token_ids'] = data['nl_token_ids'].permute(1, 0)
        data['nl_token_masks'] = data['nl_token_masks'].permute(1, 0)
        text_data = NestedTensor(data['nl_token_ids'], data['nl_token_masks'])
        text_dict = self.net(text_data=text_data, mode="language_backbone")

        # Process template (once)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
        template_att = data['template_att'].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
        template_color = template_img[:, :3, :, :]
        template_depth = template_img[:, 3:, :, :]

        color_template = self.net(img=NestedTensor(template_color, template_att), mode='backbone_color')
        depth_template = self.net(img=NestedTensor(template_depth, template_att), mode='backbone_depth')

        # Process search frames sequentially
        out_dicts = []
        gt_bboxes_seq = []

        search_images_seq = data['search_images_seq']  # List of tensors
        search_anno_seq = data['search_anno_seq']  # List of tensors

        for search_img, search_anno in zip(search_images_seq, search_anno_seq):
            search_img = search_img.view(-1, *search_img.shape[2:])  # (batch, 6, 320, 320)
            search_color = search_img[:, :3, :, :]
            search_depth = search_img[:, 3:, :, :]
            search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

            color_search = self.net(img=NestedTensor(search_color, search_att), mode='backbone_color')
            depth_search = self.net(img=NestedTensor(search_depth, search_att), mode='backbone_depth')

            # Merge and forward transformer
            color_feat_dict_list = [color_template, color_search]
            depth_feat_dict_list = [depth_template, depth_search]
            visiontext_feat_dict_list = [text_dict, color_template, color_search]

            seq_dict_color = merge_template_search(color_feat_dict_list)
            seq_dict_depth = merge_template_search(depth_feat_dict_list)
            seq_dict_vl = merge_template_search(visiontext_feat_dict_list)

            out_dict, _, _ = self.net(seq_dict_c=seq_dict_color, seq_dict_d=seq_dict_depth, seq_dict_vl=seq_dict_vl,
                                      mode="transformer", run_box_head=True, run_cls_head=False)

            out_dicts.append(out_dict)
            gt_bboxes_seq.append(search_anno)

        return out_dicts, gt_bboxes_seq

    def compute_losses_longseq(self, out_dicts, gt_bboxes_seq):
        """
        Compute accumulated loss over all search frames.
        """
        total_loss = 0.0
        total_giou_loss = 0.0
        total_l1_loss = 0.0
        total_iou = 0.0
        count = 0

        for out_dict, gt_bbox in zip(out_dicts, gt_bboxes_seq):
            # Get boxes
            pred_boxes = out_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

            # Accumulate
            total_giou_loss += giou_loss
            total_l1_loss += l1_loss
            total_iou += iou.detach().mean()
            count += 1

        # Average over sequence
        avg_giou_loss = total_giou_loss / count
        avg_l1_loss = total_l1_loss / count
        avg_iou = total_iou / count

        # weighted sum
        loss = self.loss_weight['giou'] * avg_giou_loss + self.loss_weight['l1'] * avg_l1_loss

        # status for log
        status = {"Loss/total": loss.item(),
                  "Loss/giou": avg_giou_loss.item(),
                  "Loss/l1": avg_l1_loss.item(),
                  "IoU": avg_iou.item(),
                  "SeqLen": count}

        return loss, status

