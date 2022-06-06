from tokenize import group
from torch import nn
from .bottomup import BottomUp
from .topdown import TopDownRefinement

class SoftGroup(nn.Module):

    def __init__(self,
                 in_feat_dim=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        self.bottomup = BottomUp(
            in_feat_dim,
            num_blocks,
            semantic_classes,
            ignore_label,
            fixed_modules,
            test_cfg.x4_split
        )

        if not self.semantic_only:
            self.topdown = TopDownRefinement(
                in_feat_dim,
                semantic_classes,
                instance_classes,
                sem2ins_classes,
                ignore_label,
                grouping_cfg,
                instance_voxel_cfg,
                train_cfg,
                test_cfg,
                fixed_modules
            )

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self.bottomup, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses['loss'] = loss
        for loss_name, loss_value in losses.items():
            losses[loss_name] = loss_value.item()
        return loss, losses

    def forward(self, batch, return_loss=False):
        if return_loss:
            losses = {}
            semantic_scores, pt_offsets, output_feats, pointwise_loss = self.bottomup.bottomup_forward_train(**batch)
            losses.update(pointwise_loss)
            if not self.semantic_only:
                instance_loss = self.topdown.forward_train(
                    semantic_scores=semantic_scores,
                    pt_offsets=pt_offsets,
                    output_feats=output_feats,
                    **batch)
                losses.update(instance_loss)
            return self.parse_losses(losses)
        else:
            rets = {}
            semantic_scores, pt_offsets, output_feats, bottomup_rets = self.bottomup.bottomup_forward_test(**batch)
            rets.update(bottomup_rets)
            if not self.semantic_only:
                topdown_rets = self.topdown.forward_test(
                    semantic_scores=semantic_scores,
                    pt_offsets=pt_offsets,
                    output_feats=output_feats,
                    **batch)
                rets.update(topdown_rets)
            return rets