# Training

We provide the training commands here.

## Single-node training: regular condition
As an example, to train `Swin-Tiny`, simply run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python _ET_pp_main_swinT.py \
        --tag /path/to/save/results \
        --epoch 200 \
        --seed 0
```
- In addition to `_ET_pp_main_swinT.py`, you may run other scripts for training other models, as listed in the following.
```
    _ET_pp_main_deitS.py
    _ET_pp_main_swinT.py
    _ET_pp_main_swinS.py
    _ET_pp_main_swinB.py
    _ET_pp_main_cswinT.py
    _ET_pp_main_cswinS.py
    _ET_pp_main_cswinB.py
    _ET_pp_main_res50.py
    _ET_pp_main_convnextT.py
    _ET_pp_main_convnextS.py
    _ET_pp_main_convnextB.py
```
- The default dataset directory is set with `--data_path /home/data/imagenet/` in each script, please modify it if necessary.
- `--epoch 200` corresponds to a 1.5x speedup compared to the standard 300-epoch training procedure. You may modify it for a varying number of training budgets.
- The data pre-processing speed of CPUs may be insufficient to support too many GPUs (especially for small models). A larger `--num_workers` (within each script) may alleviate this problem.
- The hyper-parameters and configurations for training different models follow from their original papers. 
- Advanced: you may consider adjusting `--batch_size`, `--update_freq`, and `--lr` simultaneously to make sure that your GPU devices have been fully utilized. The current settings are based on 8 NVIDIA 3090 GPUs


## Single-node training: limited CPU/memory capabilities
When the capabilities of CPU/memory are insufficient (even though they are fully utilized) to support the data pre-processing requirements of GPUs, you can activate the replay buffer using the scripts with `wReplay`. This technique can reduce the data pre-processing loads exponentially. Take training `DeiT-Small` as an example:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python _ET_pp_wReplay_main_deitS.py \
        --tag /path/to/save/results \
        --epoch 200 \
        --seed 0
```
- In addition to `_ET_pp_wReplay_main_deitS.py`, you may run other scripts for training other models, as listed in the following.
```
    _ET_pp_wReplay_main_deitS.py
    _ET_pp_wReplay_main_swinT.py
    _ET_pp_wReplay_main_cswinT.py
    _ET_pp_wReplay_main_res50.py
```

## Multi-node training
Please refer to [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md) for the instructions of multi-node training.


## Object detection & instance segmentation on COCO
Please refer to [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). The checkpoints obtained from this repo are fully compatible with them. 


## Semantic segmentation on ADE20K
Please refer to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). The checkpoints obtained from this repo are fully compatible with them.