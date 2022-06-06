import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from collections import OrderedDict
from .components import UBlock, BasicResBlock, MLPBlock
from segmentation.ops import ballquery_batch_p, bfs_cluster, voxelization, voxelization_idx
from ..util import cuda_cast, force_fp32, rle_encode
from tqdm import tqdm

class BottomUp(nn.Module):
    def __init__(self,
                 in_feat_dim=32,
                 num_blocks=7,
                 semantic_classes=20,
                 ignore_label=-100,
                 fixed_modules=[],
                 x4_split=False):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.num_blocks = num_blocks
        self.semantic_classes = semantic_classes
        self.ignore_label = ignore_label
        self.fixed_modules = fixed_modules
        self.x4_split = x4_split
        
        normalizer = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        channel_list = [in_feat_dim * (i + 1) for i in range(num_blocks)]
        
        # Backbone
        self.input_conv = spconv.SparseSequential(spconv.SubMConv3d(6, in_feat_dim, kernel_size=3, padding=1, bias=False, indice_key="Conv1"))
        self.unet = UBlock(channel_list, block_reps=2, block=BasicResBlock, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(normalizer(in_feat_dim), nn.ReLU())
        
        # Semantic Branch
        self.semantic_linear = MLPBlock(in_feat_dim, semantic_classes, num_layers=2)
        
        # Offset Branch
        self.offset_linear = MLPBlock(in_feat_dim, 3, num_layers=2)

        # Initialize weights
        for element in self.modules():
            if isinstance(element, MLPBlock):
                element.init_weights()
            if isinstance(element, nn.BatchNorm1d):
                nn.init.constant_(element.weight, 1)
                nn.init.constant_(element.bias, 0)

        for mod in fixed_modules:
            mod = getattr(self, mod, None)
            if mod == None:
                continue
            for param in mod.parameters():
                param.requires_grad = False

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.bottomup_forward_train(**batch)
        else:
            return self.bottomup_forward_test(**batch)

    @cuda_cast
    def bottomup_forward_train(self, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                               semantic_labels, instance_labels,
                               pt_offset_labels, spatial_shape, batch_size, **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)

        # point wise losses
        point_wise_loss = self.pointwise_loss(
            semantic_scores=semantic_scores,
            semantic_labels=semantic_labels,
            offset_vec=pt_offsets,
            offset_labels=pt_offset_labels,
            instance_labels=instance_labels
        )

        return semantic_scores, pt_offsets, output_feats, point_wise_loss

    @cuda_cast
    def bottomup_forward_test(self, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                              semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                              scan_ids, **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map, self.x4_split)
        if self.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]

        return semantic_scores, pt_offsets, output_feats, dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
            offset_labels=pt_offset_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy())
        
    def forward_backbone(self, bb_input, bb_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(bb_input, bb_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(bb_input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[bb_map.long()]
        
        semantic_score = self.semantic_linear(output_feats)
        offset_vec = self.offset_linear(output_feats)
        
        return semantic_score, offset_vec, output_feats
    
    # This code is borrowed from original SoftGroup impolementation
    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: divide and forward 4 parts of a scene."""
        outs = []
        print(x.indices)
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    # This code is borrowed from original SoftGroup impolementation
    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    # This code is borrowed from original SoftGroup impolementation
    def get_batch_offsets(self, batch_idxs, batch_size):
        batch_offsets = torch.zeros(batch_size + 1).int().contiguous().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        return batch_offsets
    
    def pointwise_loss(self, semantic_scores, semantic_labels, offset_vec, offset_labels, instance_labels):
        semantic_loss = F.cross_entropy(semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        
        indices = (instance_labels != self.ignore_label)
        if indices.sum() == 0:
            offset_loss = 0
        else:
            offset_loss = F.l1_loss(offset_vec[indices], offset_labels[indices], reduction='sum') / indices.sum()
        
        losses = {}
        losses["semantic_loss"] = semantic_loss
        losses["offset_loss"] = offset_loss

        return losses