# This code is borrowed from original SoftGroup impolementation


import sys
import argparse
import datetime
import os
import os.path as osp
import shutil
import time

import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.model.softgroup import SoftGroup
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            cosine_lr_after_step,
                            get_max_memory,
                            is_multiple, is_power2)
from tqdm import tqdm, trange
from checkpoint import realign_parameter_keys

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, train_loader, cfg, writer):
    model.train()
    meter_dict = {}

    pbar = tqdm(train_loader, desc='Batch', dynamic_ncols=True)

    for i, batch in enumerate(pbar, start=1):
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True)

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            log_dict = dict(
                epoch=f'{epoch}/{cfg.epochs}',
                batch=f'{i}/{len(train_loader)}',
                lr=f'{lr:.2g}',
                mem=get_max_memory(),
            )
            loss_dict = {k: f'{v.val:.4f}' for k, v in meter_dict.items()}
            log_dict.update(loss_dict)
            log = [f'{k}={v}' for k, v in log_dict.items()]
            tqdm.write(', '.join(log))
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, val_loader, cfg, writer):
    tqdm.write('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    progress_bar = tqdm(total=len(val_loader))
    val_set = val_loader.dataset
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch)
            results.append(result)
            progress_bar.update(1)
        progress_bar.close()
    for res in results:
        all_sem_preds.append(res['semantic_preds'])
        all_sem_labels.append(res['semantic_labels'])
        all_offset_preds.append(res['offset_preds'])
        all_offset_labels.append(res['offset_labels'])
        all_inst_labels.append(res['instance_labels'])
        if not cfg.model.semantic_only:
            all_pred_insts.append(res['pred_instances'])
            all_gt_insts.append(res['gt_instances'])
    if not cfg.model.semantic_only:
        tqdm.write('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(val_set.CLASSES)
        eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
        writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
        writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
        writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
        tqdm.write('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
            eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
    tqdm.write('Evaluate semantic segmentation and offset MAE')
    miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label)
    acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label)
    mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
                                cfg.model.ignore_label)
    writer.add_scalar('val/mIoU', miou, epoch)
    writer.add_scalar('val/Acc', acc, epoch)
    writer.add_scalar('val/Offset MAE', mae, epoch)


def main():
    print(sys.path)
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    # work_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    tqdm.write(f'Config:\n{cfg_txt}')
    tqdm.write(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = SoftGroup(**cfg.model).cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train)
    val_set = build_dataset(cfg.data.test)
    train_loader = build_dataloader(
        train_set, training=True, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, **cfg.dataloader.test)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if cfg.pretrain:
        loaded = torch.load(cfg.pretrain, map_location=torch.device("cuda:0"))['net']
        # loaded = torch.load(cfg.pretrain, map_location=torch.device("cpu"))['net']['state_dict']
        tt = []
        for k in loaded.keys():
            if 'iou_score' in k or 'mask_linear' in k:
                tt.append(k)
            if cfg.data.train.type == 's3dis' and 'semantic_linear' in k:
                tt.append(k)
        # print(tt)
        for k in tt:
            del loaded[k]
        model_state_dict = model.state_dict()
        # tt = []
        # for k in model_state_dict.keys():
        #     if 'iou_score' in k:
        #         tt.append(k)
        # print(tt)
        # print(set(k.split('.')[0] for k in loaded.keys()))
        # exit()
        # model_state_dict = strip_prefix_if_present(model_state_dict, prefix="bottomup.")
        # model_state_dict = strip_prefix_if_present(model_state_dict, prefix="topdown.")
        realign_parameter_keys(model_state_dict, loaded)
        tqdm.write(f'Load pretrain from {cfg.pretrain}')
        model.load_state_dict(model_state_dict)
        # tqdm.write(f'Load pretrain from {cfg.pretrain}')
        # load_checkpoint(cfg.pretrain, model)

    # train and val
    tqdm.write('Training')

    for epoch in trange(start_epoch, cfg.epochs + 1, desc='Epoch'):
        train(epoch, model, optimizer, train_loader, cfg, writer)
        if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)):
            validate(epoch, model, val_loader, cfg, writer)
        writer.flush()


if __name__ == '__main__':
    main()
