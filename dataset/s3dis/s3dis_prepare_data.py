from glob import glob
import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

ROOM_TYPES = {
    'conferenceRoom': 0,
    'copyRoom': 1,
    'hallway': 2,
    'office': 3,
    'pantry': 4,
    'WC': 5,
    'auditorium': 6,
    'storage': 7,
    'lounge': 8,
    'lobby': 9,
    'openspace': 10,
}

OBJECT_LABEL = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'chair': 7,
    'table': 8,
    'bookcase': 9,
    'sofa': 10,
    'board': 11,
    'clutter': 12,
}

INV_OBJECT_LABEL = {value: key for key, value in OBJECT_LABEL.items()}

def process(area, room, rootpath):
    full_path = os.path.join(rootpath, area, room, room + '.txt')
    room_info = np.ascontiguousarray(pd.read_csv(full_path, sep=' '))
    coord = room_info[:,:3].astype('float').copy()
    color = room_info[:,3:].astype('int').copy()
    color = color / 127.5 - 1
    room_type = room.split('_')[0]
    room_type_id = ROOM_TYPES[room_type]
    del room_info

    num_points = coord.shape[0]
    semantic_label = np.zeros(num_points, dtype='int64')
    instance_label = np.ones(num_points, dtype='int64') * -100
    neighbor = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coord)
    object_list = glob(os.path.join(rootpath, area, room, "Annotations", "*.txt"))

    for object_num, object_path in enumerate(object_list):
        # use NearestNeighbors object as a way of finding the indices of points in each object
        object_info = np.ascontiguousarray(pd.read_csv(object_path, sep=' '))
        _, object_point_idxs = neighbor.kneighbors(object_info[:,:3])
        
        semantic_label[object_point_idxs] = OBJECT_LABEL[object_path.split('/')[-1].split('_')[0]]
        instance_label[object_point_idxs] = object_num + 1

    return (coord, color, semantic_label, instance_label, room_type_id)

def train_val_data_process():
    if not os.path.exists("./preprocess"):
        os.mkdir("./preprocess")
    rootpath = "./Stanford3dDataset_v1.2"
    AREA_LIST = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']   

    for area in AREA_LIST:
        room_path_list = glob(os.path.join(rootpath, area, "*/*.txt"))
        for room_path in room_path_list:
            room = room_path.split('/')[-1].split('.')[0]
            coord, color, semantic_label, instance_label, room_type_id = process(area, room, rootpath)
            scene = area + "_" + room
            save_path = os.path.join("./preprocess", scene + "_inst_nostuff.pth")
            torch.save((coord, color, semantic_label, instance_label, room_type_id, scene), save_path)

def val_gt_text_s3dis():
    if not os.path.exists("./val_gt"):
        os.mkdir("./val_gt")

    val_files = sorted(glob("./preprocess/*_inst_nostuff.pth"))
    for filename in val_files:
        val_data = torch.load(filename)
        coord, color, semantic_label, instance_label, room_type_id, scene = val_data
        instance_label_new = np.zeros(coord.shape[0], dtype=np.int32)
        
        for instnum in range(1, int(instance_label.max()) + 1):
            idxs = np.where(instance_label == instnum)[0]
            if len(idxs) == 0:
                continue
            label = int(semantic_label[idxs[0]])
            if label == -100:
                label = 0
            instance_label_new[idxs] = semantic_label_idxs[label] * 1000 + instnum

        np.savetxt(os.path.join('./val_gt', scene + '.txt'), instance_label_new, fmt='%d')
        print("Saved " + scene + " val_gt file!")

train_val_data_process()
val_gt_text_s3dis()