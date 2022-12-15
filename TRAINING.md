# Training

We provide the training and fine-tuning commands here.

## Single-node Training
As an example, training `Swin-Tiny` on ImageNet-1K with a 8-GPU node:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ET_training.py \
    --data_path /path/to/imagenet-1k \
    --output_dir /path/to/save_results \
    --model swin_tiny \
    --final_bs 256 --epochs 300 \
    --num_gpus 8 --num_workers 10
```

- Using `--model` to specify the type of model to be trained. Currently, training ResNet-50, ConvNeXt, DeiT, Swin and CSWin is supported. See [ET_training.py](ET_training.py) for there names.
- Using `CUDA_VISIBLE_DEVICES` and `--num_gpus` to set the number of GPUs to be used. 
- The argument `--final_bs` refers to the number of samples on each GPU at a time, and it corresponds to the original inputs at the final stage of training (*i.e.*, `B=224`). For the inputs at earlier training stages (*i.e.*, `B=160/192`), this number is increased for higher training efficiency. Importantly, the effective batch size (*i.e.*, GPU number * batch size * update frequency) is fixed to be `4096`. See [ET_training.py](ET_training.py) for the details.
- The data pre-processing speed of CPUs may be insufficient to support too many GPUs (especially for small models). A larger `--num_workers` may alleviate this problem
- The hyper-parameters and configurations for training different models follows from their original papers. They are specified in [ET_training.py](ET_training.py).



## Multi-node Training
Please refer to [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md) for the instructions of multi-node training.


## Object Detection & Instance Segmentation on COCO
Please refer to [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). The checkpoints obtained from this repo is fully compatible with them. In particular, we give an example for configuring the environments:
```
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git swinTxs
cd swinTxs
python setup.py develop # run this in the Swin-Transformer-Object-Detection main directory

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmpycocotools
pip install mmdet
```