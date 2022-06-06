#!/usr/bin/env bash

python train.py configs/softgroup_s3dis_fold5.yaml ${@:1} --skip_validate
# python train.py configs/softgroup_s3dis_fold5.yaml ${@:1} --skip_validate