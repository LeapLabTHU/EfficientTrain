# Pre-trained Models & Evaluation

Here we provide the pre-trained models and the evaluation instructions.


## ImageNet-1K pre-trained models

These models are also available at [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d214a1f392684840834b/).

| Model | #Param |  #FLOPs | acc@1 | #epochs | Training Speedup | link |
|:---:|:---:|:---:|:---:| :---:| :---:| :---:|
| ConvNeXt-Tiny  | 29M | 4.5G  | 82.2% | 300 | 1.49x | [Google Drive](https://drive.google.com/file/d/15FjPFICnhzq7eeNB-_y7ROLfzHMGAN05/view?usp=sharing) |
| ConvNeXt-Small | 50M | 8.7G  | 83.2% | 300  | 1.50x | [Google Drive](https://drive.google.com/file/d/13r0hUXDihEghi2_sM9vx3Nyqz0np6lBL/view?usp=sharing) |
| ConvNeXt-Base  | 89M | 15.4G | 83.8% | 300  | 1.48x | [Google Drive](https://drive.google.com/file/d/1LcM4rRFWsLIMbopA3UkmW12dIrdE0Adi/view?usp=sharing) |
| DeiT-Tiny     | 5M | 1.3G  | 73.3% | 300  | 1.55x | [Google Drive](https://drive.google.com/file/d/1hbzSIcHyDBaydlAIjgkJE7Bgua-YTsYO/view?usp=sharing) |
| DeiT-Tiny     | 5M | 1.3G  | 74.4% | 450  | 1.04x | [Google Drive](https://drive.google.com/file/d/16jRohjEM4eK8W1mxrUy4Pd5WZMNaqQ0B/view?usp=sharing) |
| DeiT-Small     | 22M | 4.6G  | 80.4% | 300  | 1.51x | [Google Drive](https://drive.google.com/file/d/1WB3MgIvnsPrrFXglIHvNck3ohYzxbdd5/view?usp=sharing) |
| DeiT-Small     | 22M | 4.6G  | 81.0% | 450  | 1.01x | [Google Drive](https://drive.google.com/file/d/1ro_KGEoP10d-QIVoIb7Dacnn47i8NPM3/view?usp=sharing) |
| Swin-Tiny      | 28M | 4.5G  | 81.4% | 300  | 1.49x | [Google Drive](https://drive.google.com/file/d/13bCG-1E5YnfVm8goFKBRFedYwAyUjZwB/view?usp=sharing) |
| Swin-Small     | 50M | 8.7G  | 83.2% | 300  | 1.50x | [Google Drive](https://drive.google.com/file/d/1XbA8hhTKT1BfB1NhsiHTfXZFoq59MTWj/view?usp=sharing) |
| Swin-Base      | 88M | 15.4G | 83.6% | 300  | 1.50x | [Google Drive](https://drive.google.com/file/d/1Hq1IVcnpKI3PvjyYOY7Ik05vHOIul34Y/view?usp=sharing) |



## Evaluation
We give an example evaluation command for `Swin-Tiny`:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --model swin_tiny --drop_path 0.2 \
    --eval true --batch_size 128 --input_size 224 \
    --data_path /path/to/imagenet-1k \
    --resume /path/to/checkpoint
```

This should yield 
```
* Acc@1 81.380 Acc@5 95.602 loss 0.809
```

- For other models, please change `--model` and `--resume` accordingly. You can get the pre-trained models from the table above. 
- Setting a model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in `timm` behaves the same during evaluation, but it is required in training. See [ET_training.py](ET_training.py) for the values used for different models.
