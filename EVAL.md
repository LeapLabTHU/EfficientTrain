# Pre-trained Models & Evaluation

Here we provide the pre-trained models and the evaluation instructions.


## ImageNet-1K pre-trained models

| Model | #Param |  #FLOPs | acc@1 | Training Speedup | link |
|:---:|:---:|:---:|:---:| :---:| :---:|
| ConvNeXt-Tiny  | 29M | 4.5G  | 82.2% | 1.49x | [Google Drive](https://drive.google.com/file/d/15FjPFICnhzq7eeNB-_y7ROLfzHMGAN05/view?usp=share_link) |
| ConvNeXt-Small | 50M | 8.7G  | 83.2% | 1.50x | [Google Drive](https://drive.google.com/file/d/13r0hUXDihEghi2_sM9vx3Nyqz0np6lBL/view?usp=share_link) |
| ConvNeXt-Base  | 89M | 15.4G | 83.8% | 1.48x | [Google Drive](https://drive.google.com/file/d/1LcM4rRFWsLIMbopA3UkmW12dIrdE0Adi/view?usp=share_link) |
| DeiT-Small     | 22M | 4.6G  | 80.4% | 1.51x | [Google Drive](https://drive.google.com/file/d/1WB3MgIvnsPrrFXglIHvNck3ohYzxbdd5/view?usp=share_link) |
| Swin-Tiny      | 28M | 4.5G  | 81.3% | 1.49x | [Google Drive](https://drive.google.com/file/d/13bCG-1E5YnfVm8goFKBRFedYwAyUjZwB/view?usp=share_link) |
| Swin-Small     | 50M | 8.7G  | 83.2% | 1.50x | [Google Drive](https://drive.google.com/file/d/1XbA8hhTKT1BfB1NhsiHTfXZFoq59MTWj/view?usp=share_link) |
| Swin-Base      | 88M | 15.4G | 83.6% | 1.50x | [Google Drive](https://drive.google.com/file/d/1Hq1IVcnpKI3PvjyYOY7Ik05vHOIul34Y/view?usp=share_link) |



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
* Acc@1 81.338 Acc@5 95.516 loss 0.805
```

- For other models, please change `--model` and `--resume` accordingly. You can get the pre-trained models from the table above. 
- Setting a model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in `timm` behaves the same during evaluation, but it is required in training. See [ET_training.py](ET_training.py) for the values used for different models.
