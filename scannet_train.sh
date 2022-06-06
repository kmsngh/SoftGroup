#!/usr/bin/env bash

python train.py configs/softgroup_scannet.yaml ${@:1} --skip_validate
