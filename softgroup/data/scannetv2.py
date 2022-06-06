import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from segmentation.ops import voxelization_idx

class ScanNetV2Dataset(Dataset):
    
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
               'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
    
    def __init__(self,
                 rootpath,
                 train_or_val,
                 voxel_cfg=None,
                 training=True,
                 repeat=1,
                 **kwargs):
        self.dirpath = os.path.join(rootpath, train_or_val)
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.filelist = sorted(glob(os.path.join(self.dirpath, "*_inst_nostuff.pth")) * self.repeat)
    
    def __len__(self):
        return len(self.filelist)
    
    def instance_info(self, coord, instance_label, semantic_label):
        offset_vec = np.ones((coord.shape[0], 3)) * -100
        instance_pointnum = []
        instance_cls = []
        
        # find total number of instances
        instance_num = int(instance_label.max()) + 1
        
        for instnum in range(instance_num):
            # find how many points are in each instance(instance_pointnum)
            idxs = np.where(instance_label == instnum)[0]
            instance_pointnum.append(len(idxs))
            
            # find center of each instance using mean(0) method, update offset_vec
            mean_vec = coord[idxs].mean(0)
            offset_vec[idxs] = mean_vec - coord[idxs]
            
            # find which classification current instance belongs to
            instance_cls.append(semantic_label[idxs[0]])
        instance_cls = [x - 2 if x != -100 else x for x in instance_cls]
        
        return (instance_num, instance_pointnum, instance_cls, offset_vec)
    
    def __getitem__(self, index):
        filename = self.filelist[index]
        coord, color, semantic_label, instance_label = torch.load(filename)
        scan_id = os.path.basename(filename).replace("_inst_nostuff.pth", '')
        inst_info = self.instance_info(coord, instance_label.astype(np.int32), semantic_label)
        instance_num, instance_pointnum, instance_cls, offset_vec = inst_info
        coord_float = coord.copy()
        coord = coord * self.voxel_cfg.scale
        coord = coord - coord.min(0)
        
        coord = torch.from_numpy(coord).long()
        coord_float = torch.from_numpy(coord_float)
        feat = torch.from_numpy(color).float()
        
        # small jitter, for data augmentation
        if self.training:
            feat = feat + torch.randn(3) * 0.1
            
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        offset_vec = torch.from_numpy(offset_vec)
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, instance_num,
                instance_pointnum, instance_cls, offset_vec)
    
    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []
        
        total_instance_num = 0
        cur_batch_idx = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label, instance_label, instance_num, instance_pointnum_cur, 
             instance_cls_cur, offset_vec) = data
            # scan_ids
            scan_ids.append(scan_id)
            # coord (append batch index)
            idxcolumn = torch.ones(coord.shape[0], 1) * cur_batch_idx
            coord = torch.cat((idxcolumn, coord), 1)
            coords.append(coord)
            # coord_float
            coords_float.append(coord_float)
            # feat
            feats.append(feat)
            # semantic_label
            semantic_labels.append(semantic_label)
            # instance_label (update instance numbers)
            instance_label[np.where(instance_label != -100)[0]] += total_instance_num
            total_instance_num += instance_num
            instance_labels.append(instance_label)
            # instance_pointnum
            instance_pointnum += instance_pointnum_cur
            # instance classification id
            instance_cls += instance_cls_cur
            # offset vector
            pt_offset_labels.append(offset_vec)
            # current batch index increase
            cur_batch_idx += 1
        
        # merge all data in batch to single tensor/list
        coords = torch.cat(coords, 0)
        batch_idxs = coords[:,0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)
        feats = torch.cat(feats, 0)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        instance_labels = torch.cat(instance_labels, 0).long()
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)
        pt_offset_labels = torch.cat(pt_offset_labels, 0).float()
        
        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords.long(), cur_batch_idx)
        
        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': cur_batch_idx,
        }