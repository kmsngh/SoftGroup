from glob import glob
import json
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import os

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
]

# Author uses map of size 150, but since there are only 41 types of items in nyu40 labels, we used 41
map_id = np.ones(41) * -100
for i, x in enumerate(semantic_label_idxs):
    map_id[x] = i

def rawcategory_to_nyu40_func():
    rawcategory_to_nyu40 = {}
    label_set = set(['unannotated'] + semantic_label_names)
    f = open("scannetv2-labels.combined.tsv")
    f.readline()
    for line in f.readlines():
        line = line.strip('\n').strip().split('\t')
        raw_category = line[1]
        nyu40 = line[7]
        if nyu40 not in label_set:
            rawcategory_to_nyu40[raw_category] = 'unannotated'
        else:
            rawcategory_to_nyu40[raw_category] = nyu40
    f.close()
    return rawcategory_to_nyu40

def process(clean2ply: str, clean2labelsply: str, segsjson: str, aggjson: str):
    rawcategory_to_nyu40 = rawcategory_to_nyu40_func()
    plydata = PlyData.read(clean2ply)
    labeldata = PlyData.read(clean2labelsply)
    segs = json.load(open(segsjson))
    agg = json.load(open(aggjson))
    
    # xyz coordinates and rgb color data
    plydata = np.array([list(x) for x in plydata.elements[0]])
    coord = plydata[:,:3] - plydata[:,:3].mean(0)
    color = plydata[:,3:] / 127.5 - 1
    
    # convert original nyu40 labels to 0 ~ 19, or -100 to be used as semantic label
    labeldata = np.array(labeldata.elements[0]['label'])
    semantic_label = map_id[labeldata]
    
    # find instance labels
    instance_label = np.ones(plydata.shape[0]) * -100
    segid_to_points = {}
    for pointid, segid in enumerate(segs["segIndices"]):
        if segid not in segid_to_points:
            segid_to_points[segid] = []
        segid_to_points[segid].append(pointid)
    for instancenum, data in enumerate(agg["segGroups"]):
        list_of_segid = data["segments"]
        for segid in list_of_segid:
            instance_label[segid_to_points[segid]] = instancenum
    
    return coord, color, semantic_label, instance_label, segs["sceneId"]

def train_val_data_process():
    clean2ply_list = sorted(glob("./scans/scene*_*/scene*_*_vh_clean_2.ply"))
    clean2labelsply_list = sorted(glob("./scans/scene*_*/scene*_*_vh_clean_2.labels.ply"))
    segsjson_list = sorted(glob("./scans/scene*_*/scene*_*_vh_clean_2.0.010000.segs.json"))
    aggjson_list = sorted(glob("./scans/scene*_*/scene*_*.aggregation.json"))
    
    train = open("scannetv2_train.txt", 'r')
    val = open("scannetv2_val.txt", 'r')
    train_list = []
    val_list = []
    for line in train.readlines():
        train_list.append(line.strip('\n').strip())
    for line in val.readlines():
        val_list.append(line.strip('\n').strip())
    train_list = set(train_list)
    val_list = set(val_list)
    
    if not os.path.exists("./train"):
        os.mkdir("./train")
    if not os.path.exists("./val"):
        os.mkdir("./val")
    if not os.path.exists("./test"):
        os.mkdir("./test")

    # save coord, color, semantic label, instance label as pth file
    for i in range(len(clean2ply_list)):
        coord, color, semantic_label, instance_label, scene = process(clean2ply_list[i], clean2labelsply_list[i], 
                                                                      segsjson_list[i], aggjson_list[i])
        if scene in train_list:
            torch.save((coord, color, semantic_label, instance_label, scene), "./train/" + scene + "_inst_nostuff.pth")
            print(f"Num: {i} / Saved processed pth file for {scene} in train")
        elif scene in val_list:
            torch.save((coord, color, semantic_label, instance_label, scene), "./val/" + scene + "_inst_nostuff.pth")
            print(f"Num: {i} / Saved processed pth file for {scene} in val")
        else:
            raise Exception("Scene does not belong to train, test, val!")

def val_gt_text_scannetv2():
    if not os.path.exists("./val_gt"):
        os.mkdir("./val_gt")

    val_files = sorted(glob("./val/*_inst_nostuff.pth"))
    for filename in val_files:
        val_data = torch.load(filename)
        coord, color, semantic_label, instance_label, scene = val_data
        instance_label_new = np.zeros(coord.shape[0], dtype=np.int32)
        
        for instnum in range(int(instance_label.max()) + 1):
            idxs = np.where(instance_label == instnum)[0]
            label = int(semantic_label[idxs[0]])
            if label == -100:
                label = 0
            instance_label_new[idxs] = semantic_label_idxs[label] * 1000 + instnum + 1

        np.savetxt(os.path.join('./val_gt', scene + '.txt'), instance_label_new, fmt='%d') 

train_val_data_process()
val_gt_text_scannetv2()