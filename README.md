# Remote Sensing Image Change Detection with Transformers

Here, we provide the pytorch implementation of the paper: Relation-Decomposition Change Enhancement Block via Supervised Contrastive Learning for Remote Sensing Change Detection.

For more information, please see our published paper later. 


## Requirements

```
conda create -n ceblock python=3.9
conda activate ceblock
pip install -r requirements.txt
```

## Installation

Clone this repo:

```shell
git https://github.com/Git-hub-Xin/CEBlock.git
cd CDBlock
```

## Checkpoint

We have provided some model weight files trained on the WHU-CD dataset.

Firstly, you can download our CEBlock pretrained model. After downloaded the pretrained model, you can put it in `checkpoints/CEBlock/`. 
Please replace `checkpoint_name=/home/best_ckpt.pt in the scripts/eval.sh` file with the actual path to the weight file.

Backbone | Base  | CEBlock 
-- | -- | -- 
R18 | [weight](https://github.com/Git-hub-Xin/CEBlock/releases/download/v1.0/base_r18_best_ckpt.pt) | [weight](https://github.com/Git-hub-Xin/CEBlock/releases/download/v1.0/ceblock_r18_best_ckpt.pt)
R34 | [weight](https://github.com/Git-hub-Xin/CEBlock/releases/download/v1.0/base_r34_best_ckpt.pt) | -
R50 | [weight](https://github.com/Git-hub-Xin/CEBlock/releases/download/v1.0/base_r50_best_ckpt.pt) | -
R101 | [weight](https://github.com/Git-hub-Xin/CEBlock/releases/download/v1.0/base_r101_best_ckpt.pt) | _

Then, run a test to get started as follows:

```python
bash scripts/eval.sh 
```

After that, you can find the prediction results in `samples/predict`.

## Train

You can find the training script `run_cd.sh` in the folder `scripts`. You can run the script file by `bash scripts/run_cd.sh` in the command environment.

The detailed script file `run_cd.sh` is as follows:

```cmd
gpus=0
checkpoint_root=checkpoints
data_name=WHU
img_size=256
batch_size=2
lr=0.01
max_epochs=200
net_G=base_transformer_pos_s4_dd8
lr_policy=linear
split=train
split_val=val
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}
```

## Evaluate

You can find the evaluation script `eval.sh` in the folder `scripts`. You can run the script file by `bash scripts/eval.sh` in the command environment.

The detailed script file `eval.sh` is as follows:

```cmd
gpus=0
data_name=WHU
net_G=base_transformer_pos_s4_dd8
split=test
project_name=CEBlock
checkpoint_name=/home/best_ckpt.pt

python ../eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Data Download 

SYSU-CD: https://github.com/liumency/SYSU-CD

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

DSIFN-CD: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```

```
