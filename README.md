# Installation
## Requirements
The following describes the environment we used:
- `python=3.7`
- `pytorch=1.11`
- `cuda=11.3`

## Install packages
```
pip install spconv-cu113      # change if cuda version is different
pip install requirements.txt
```

## Download Dataset

### ScanNet v2
You can find instructions to download ScanNet v2 [here](http://www.scan-net.org/).

Copy the `scans` directory into `dataset/scannetv2`.

### S3DIS

You can find instructions to download S3DIS [here](http://buildingparser.stanford.edu/dataset.html).

Copy the `Stanford3dDataset_v1.2` directory into `dataset/s3dis`.

# Train

This model uses a pre-trained backbone, which is a [HAIS](https://github.com/hustvl/HAIS) model trained on ScanNet semantic segmentation.
You can find the pretrained backbone model provided by the authors of SoftGroup [here](https://drive.google.com/file/d/1FABsCUnxfO_VlItAzDYAwurdfcdK-scs/view) and place it in the project's main directory.

## ScanNet v2

To train on ScanNet v2, the backbone parameters are frozen.
You can train by running `bash scannet_train.sh`.

## S3DIS

To train on S3DIS, the backbone needs to be trained by running `bash s3dis_pretrain.sh` and the model is finetuned by running `bash s3dis_finetune.sh`.

# Evalulate
You can find our pretrained model trained on the ScanNet v2 dataset [here](https://drive.google.com/drive/folders/1XSi5SWwUtet_4Wtr2UD_u3cWbZSg2DGr?usp=sharing).

Run `python test.py configs/{model config}.yaml {path_to_model}` to evaluate the model.