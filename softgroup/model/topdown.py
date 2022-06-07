import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import UBlock
from .components import BasicResBlock
from .components import MLPBlock
import spconv.pytorch as spconv
import functools
from segmentation.ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from tqdm import tqdm


class TopDownRefinement(nn.Module):
    def __init__(self,
                 in_feat_dim: int=32,
                 semantic_classes: int=20,
                 instance_classes: int=18,
                 sem2ins_classes: list=[],
                 ignore_label: int=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules: list=[]):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        
        # Top-down UNet
        self.tiny_unet = UBlock([in_feat_dim, 2 * in_feat_dim], 2, BasicResBlock, indice_key_id=11)
        self.tiny_unet_outputlayer = spconv.SparseSequential(functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)(in_feat_dim), nn.ReLU())

        # Classification branch
        self.cls_linear = nn.Linear(in_feat_dim, instance_classes + 1)

        # Segmentation branch
        self.mask_linear = MLPBlock(in_feat_dim, instance_classes + 1, normalizer=None, num_layers=2)

        # Mask scoring branch
        self.iou_score_linear = nn.Linear(in_feat_dim, instance_classes + 1)

        # Initialize weights
        for m in [self.cls_linear, self.iou_score_linear]:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

        for mod in fixed_modules:
            mod = getattr(self, mod, None)
            if mod == None:
                continue
            for param in mod.parameters():
                param.requires_grad = False
    
    @cuda_cast
    def forward_train(self, batch_idxs, coords_float, pt_offsets, output_feats,
                        instance_labels, instance_pointnum, instance_cls,
                        semantic_scores, **kwargs):

        proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                        batch_idxs, coords_float,
                                                        self.grouping_cfg)
        if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
            proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
            proposals_idx = proposals_idx[:proposals_offset[-1]]
        inst_feats, inst_map = self.clusters_voxelization(
            proposals_idx,
            proposals_offset,
            output_feats,
            coords_float,
            random_noise=True,
            **self.instance_voxel_cfg)
        instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
            inst_feats, inst_map)
        instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                            proposals_offset, instance_labels, instance_pointnum,
                                            instance_cls, instance_batch_idxs)

        return instance_loss

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        losses = {}
        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(
            proposals_idx,
            proposals_offset,
            instance_labels,
            instance_pointnum
        )

        foreground_idxs = (self.ignore_label != instance_cls)
        foreground_instance_cls = instance_cls[foreground_idxs]
        foreground_ious_on_cluster = ious_on_cluster[:, foreground_idxs]
        max_iou, gt_inds = foreground_ious_on_cluster.max(1)
        pos_inds = self.train_cfg.pos_iou_thr <= max_iou 
        pos_gt_inds = gt_inds[pos_inds]

        labels = foreground_instance_cls.new_full((foreground_ious_on_cluster.size(0), ), self.instance_classes)
        labels[pos_inds] = foreground_instance_cls[pos_gt_inds]
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # compute mask loss
        # ---------------------Ablation mask branch---------------------
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss
        # ---------------------Ablation mask branch---------------------

        # compute iou score loss
        # ---------------------Ablation mask score branch---------------------
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        foreground_ious = ious[:, foreground_idxs]
        gt_ious, _ = foreground_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss
        # ---------------------Ablation mask score branch---------------------
        return losses

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)

        return instance_batch_idxs, cls_scores, iou_scores, mask_scores

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        batch_offset = torch.cumsum(torch.bincount(x.indices[:, 0]), dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[x.indices[:, 0].long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(
            self,
            scan_id,
            proposals_idx,
            semantic_scores,
            cls_scores,
            iou_scores,
            mask_scores):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)    # No ablation on mask score
                # score_pred = cur_cls_scores                               # Ablation on mask score
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()         # No ablation on mask
                # cur_proposals_idx = proposals_idx.long()                  # Ablation on mask
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.cls_score_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        mask_pred = torch.cat(mask_pred_list).cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        return batch_offsets

    @cuda_cast
    def forward_test(self, batch_idxs, coords_float,
                     semantic_labels, instance_labels,
                     scan_ids, semantic_scores, pt_offsets, output_feats, **kwargs):
        
        proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                batch_idxs, coords_float,
                                                                self.grouping_cfg)
        inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                            output_feats, coords_float,
                                                            **self.instance_voxel_cfg)
        _, cls_scores, iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map)
        pred_instances = self.get_instances(scan_ids[0], proposals_idx, semantic_scores,
                                            cls_scores, iou_scores, mask_scores)
        gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
        return dict(pred_instances=pred_instances, gt_instances=gt_instances)

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        # # ----------------for ablation study where tau=none---------------- #
        # temp = torch.zeros(semantic_scores.shape)
        # argmax_indices = semantic_scores.argmax(dim=1)
        # semantic_scores = F.one_hot(argmax_indices)
        # # ----------------for ablation study where tau=none---------------- #

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                               radius, mean_active)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                                                          start_len.cpu(), npoint_thr, class_id)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)

        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        return proposals_idx, proposals_offset

    @force_fp32(apply_to='feats')
    def clusters_voxelization(
        self,
        proposal_idx,
        proposal_offset,
        pointwise_feats,
        xyz_coords,
        scale,
        spatial_shape,
        random_noise=False):
        
        c_idxs = proposal_idx[:, 1].cuda()
        xyz_coords = xyz_coords[c_idxs.long()]
        batch_idx = proposal_idx[:, 0].cuda().long()
        pointwise_feats = pointwise_feats[c_idxs.long()]

        coords_min = sec_min(xyz_coords, proposal_offset.cuda())
        coords_max = sec_max(xyz_coords, proposal_offset.cuda())

        clusters_scale = torch.clamp(1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01, max=scale)

        coords_max = coords_max * clusters_scale[:, None]
        coords_min = coords_min * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        xyz_coords = xyz_coords * clusters_scale[:, None]

        if random_noise:
            # after this, xyz_coords.long() will have some randomness
            range = spatial_shape + coords_min - coords_max
            coords_min -= torch.clamp(range + 0.001, max=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(range - 0.001, min=0) * torch.rand(3).cuda()
        xyz_coords -= coords_min[batch_idx]
        xyz_coords = xyz_coords.long()
        xyz_coords = torch.cat([proposal_idx[:, 0].view(-1, 1).long(), xyz_coords.cpu()], 1)

        inst_feats, inst2point_feats, point2inst_feats = voxelization_idx(xyz_coords, int(proposal_idx[-1, 0]) + 1)
        out_feats = voxelization(pointwise_feats, point2inst_feats.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(
            out_feats,
            inst_feats.int().cuda(),
            spatial_shape,
            int(proposal_idx[-1, 0]) + 1,
        )
        return voxelization_feats, inst2point_feats