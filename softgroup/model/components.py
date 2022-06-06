import torch
import functools
from torch import nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from collections import OrderedDict

class MLPBlock(nn.Sequential):

    def __init__(self, in_in_feat_dim, out_in_feat_dim, normalizer=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1), num_layers=2):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_in_feat_dim, in_in_feat_dim))
            if normalizer:
                layers.append(normalizer(in_in_feat_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_in_feat_dim, out_in_feat_dim))
        return super().__init__(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class CustomSubm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

class BasicResBlock(SparseModule):

    def __init__(self, in_in_feat_dim, out_in_feat_dim, normalizer=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1), indice_key=None):
        super().__init__()

        if in_in_feat_dim != out_in_feat_dim:
            self.i_branch = spconv.SparseSequential(
                CustomSubm3d(in_in_feat_dim, out_in_feat_dim, kernel_size=1, bias=False))
        else:
            self.i_branch = spconv.SparseSequential(nn.Identity())

        self.conv_branch = spconv.SparseSequential(
            normalizer(in_in_feat_dim), nn.ReLU(),
            spconv.SubMConv3d(
                in_in_feat_dim,
                out_in_feat_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key), normalizer(out_in_feat_dim), nn.ReLU(),
            spconv.SubMConv3d(
                out_in_feat_dim,
                out_in_feat_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        unet_blocks = OrderedDict({
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        })

        normalizer = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        self.blocks = spconv.SparseSequential(unet_blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                normalizer(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                normalizer(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    normalizer,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output