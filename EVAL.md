# Pre-trained Models & Evaluation & Fine-tuning

Here we provide the pre-trained models and the evaluation/fine-tuning instructions.


## ImageNet-1K trained models

These models are also available at [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f11eb6a2cd2d4e049fe2/).

| Model | #Param |  #FLOPs | Acc@1 | Training Speedup | #Equivalent Epochs | link |
|:---:|:---:|:---:|:---:| :---:| :---:| :---:|
| ResNet-50      | 26M | 4.1G  | 79.7% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1cuSDCD_zCXoXQ5gYkNt79fZ0Kpp4eovs/view?usp=sharing) |
| ConvNeXt-Tiny  | 29M | 4.5G  | 82.2% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/15kTX6v5UfBQ_0FJUSEKDJKouSyH2mgCe/view?usp=sharing) |
| ConvNeXt-Small | 50M | 8.7G  | 83.2% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1vzmGWDQVy44Y-WyN4O4PhwIfVPp7H7_V/view?usp=sharing) |
| ConvNeXt-Base  | 89M | 15.4G | 83.8% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1Tn3hCtHaqx2p1tN16oB0Vc5A6m_EAY-r/view?usp=sharing) |
| DeiT-Tiny      | 5M  | 1.3G  | 72.5% | ~3.0x | 100 | [Google Drive](https://drive.google.com/file/d/13bfLB3wT2W1Wnq1_nobftoH1fBwnZJ3l/view?usp=sharing) |
|                |     |       | 73.4% | ~2.0x | 150 | [Google Drive](https://drive.google.com/file/d/1M9yqjW6hDl85YmPvdBiiHyIqR7LdgTPn/view?usp=sharing) |
|                |     |       | 73.8% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/15PEpUqfmph302soGj2vC0oZr47iL5e8Z/view?usp=sharing) |
|                |     |       | 74.4% | ~1.0x | 300 | [Google Drive](https://drive.google.com/file/d/1VIx457b8MpZTsFdrezfrzjLbYTtKx07Y/view?usp=sharing) |
| DeiT-Small     | 22M | 4.6G  | 79.9% | ~3.0x | 100 | [Google Drive](https://drive.google.com/file/d/1jTjGi_alKL3OiDzuz4L_bLTQ2O59QqsS/view?usp=sharing) |
|                |     |       | 80.6% | ~2.0x | 150 | [Google Drive](https://drive.google.com/file/d/1nNsTDPkGS-05JMxYUwb77_YI5Txqq5Z6/view?usp=sharing) |
|                |     |       | 81.0% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1yKpCeDPFM9TePOXAYdxDb6A9lid7DU4x/view?usp=sharing) |
|                |     |       | 81.4% | ~1.0x | 300 | [Google Drive](https://drive.google.com/file/d/1Ob0Gh9QnVfPxlbMO2cPtos9Bf2Uf8yvp/view?usp=sharing) |
| Swin-Tiny      | 28M | 4.5G  | 80.9% | ~3.0x | 100 | [Google Drive](https://drive.google.com/file/d/1HS_c80fF0FepwcgpevB5nF5ecI2H8ytx/view?usp=sharing) |
|                |     |       | 81.4% | ~2.0x | 150 | [Google Drive](https://drive.google.com/file/d/1id4lWtee4P15vTk8DwF_AiUtrLTRjtNo/view?usp=sharing) |
|                |     |       | 81.6% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1ObG4gc_eQoFAjtTZBP2RBuyrr6h_MI1v/view?usp=sharing) |
| Swin-Small     | 50M | 8.7G  | 82.8% | ~3.0x | 100 | [Google Drive](https://drive.google.com/file/d/1Fr5-CBDypd7bIgq39q7NJ7XqOPpbh1Sv/view?usp=sharing) |
|                |     |       | 83.1% | ~2.0x | 150 | [Google Drive](https://drive.google.com/file/d/1DFSiOdxCLawYbT00Mi7faimEy3vlmcvo/view?usp=sharing) |
|                |     |       | 83.2% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/18rfw45gRlS2QZfD1r31QYnippODPP_CS/view?usp=sharing) |
| Swin-Base      | 88M | 15.4G | 83.3% | ~3.0x | 100 | [Google Drive](https://drive.google.com/file/d/1aAnY6-JwfJvs9hDov6--uVEEat4Sz26X/view?usp=sharing) |
|                |     |       | 83.5% | ~2.0x | 150 | [Google Drive](https://drive.google.com/file/d/12V8cXHRyEPitr7NBP5qzzUedUbP3xQ3q/view?usp=sharing) |
|                |     |       | 83.6% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1a_4yTFLLbS82TwCTU0Ihhy5HqAKhDMgV/view?usp=sharing) |
| CSWin-Tiny     | 23M | 4.3G  | 82.9% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1dNIpKm7bN7WKz-xrSh5tzaeCp6rnUtEf/view?usp=sharing) |
| CSWin-Small    | 35M | 6.9G  | 83.6% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1eP1CfYa_IpUSMzjfR-K6mm9kpgMNCs5s/view?usp=sharing) |
| CSWin-Base     | 78M | 15.0G | 84.3% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1psRTFl5imHcDVMNp5M4EhwbE_w6SVztB/view?usp=sharing) |
| CAFormer-S18   | 26M | 4.1G  | 83.4% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1vxh-N0ZSdR6cPQHf6wi9PXwMSd8qo1_B/view?usp=sharing) |
| CAFormer-S36   | 39M | 8.0G  | 84.3% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1ObZsgyAJ-j2L8bsx2VSzYh1LKMFV56KG/view?usp=sharing) |
| CAFormer-M36   | 56M | 13.2G | 85.0% | ~1.5x | 200 | [Google Drive](https://drive.google.com/file/d/1HVos4MVtwcbyZQAANb1jQ7E7dp5tn9LI/view?usp=sharing) |


## ImageNet-22K -> ImageNet-1K fine-tuned models

These models are also available at [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/329723de28c149d0a674/).

| Model | #Param |  #FLOPs | Acc@1 | Pre-training Speedup | link |
|:---:|:---:|:---:|:---:| :---:| :---:|
| CSWin-Base-224  | 78M | 15.0G  | 86.1% | ~3.0x  | [Google Drive](https://drive.google.com/file/d/19JrP1o4NEOjTvjzosvjnc33nATtNl2WO/view?usp=sharing) |
|                 |     |        | 86.3% | ~2.0x  | [Google Drive](https://drive.google.com/file/d/1LrmpusMIihB0sSHv6BA52bfVCLqxcfhZ/view?usp=sharing) |
| CSWin-Base-384  | 78M | 47.0G  | 87.1% | ~3.0x  | [Google Drive](https://drive.google.com/file/d/1eqTgAfM5paGcn6EB32P5OPyKvE-C4LTT/view?usp=sharing) |
|                 |     |        | 87.4% | ~2.0x  | [Google Drive](https://drive.google.com/file/d/1-yfo_nIg1ftkgV02v8HC4lKxPnyLQdFw/view?usp=sharing) |
| CSWin-Large-224 | 173M | 31.5G  | 86.9% | ~3.0x  | [Google Drive](https://drive.google.com/file/d/16n9AyG_qvY2ZcTCX_GiIN4iwBvf6Iiaj/view?usp=sharing) |
|                 |      |        | 87.1% | ~2.0x  | [Google Drive](https://drive.google.com/file/d/1FkHvZr-TFHkfk_g4rIcbeJ6RMvIQWSFG/view?usp=sharing) |
| CSWin-Large-384 | 173M | 96.8G  | 87.9% | ~3.0x  | [Google Drive](https://drive.google.com/file/d/1lFQKRQJS5610W5EaLRezgzQpoDvrem26/view?usp=sharing) |
|                 |      |        | 88.1% | ~2.0x  | [Google Drive](https://drive.google.com/file/d/15pLG8nL1BkBmaBBhIIyYKHjsiqI5eRoI/view?usp=sharing) |




## Evaluation
We give an example command for evaluating `Swin-Tiny`:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=12345 main_buffer.py \
    --model swin_tiny --drop_path 0.0 \
    --eval true --batch_size 128 --input_size 224 \
    --data_path /path/to/imagenet-1k \
    --resume /path/to/checkpoint/ET_pp_200ep_swinT.pth
```
This should yield 
```
* Acc@1 81.626 Acc@5 95.694 loss 0.785
```

- For other models, please change `--model`, `--resume`, and `--input_size` accordingly. You can get the pre-trained models from the tables above. 
- Setting a model-specific `--drop_path` is not required in evaluation, as the `DropPath` module in `timm` behaves the same during evaluation, but it is required in training.






## ImageNet-22K pre-trained models

These models are also available at [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/0110833a65c746778345/).

| Model | #Param |  #FLOPs | Pre-training Speedup | link |
|:---:|:---:|:---:|:---:|:---:|
| CSWin-Base-224  | 78M  | 15.0G | ~3.0x  | [Google Drive](https://drive.google.com/file/d/1yf-uXmRNcd1R2m-eMPLu-a8hKPS_kt1Z/view?usp=sharing) |
|                 |      | 15.0G | ~2.0x  | [Google Drive](https://drive.google.com/file/d/1WvVPVpjDtI0puMEDkJ-BT72ehq2bdBYE/view?usp=sharing) |
| CSWin-Large-224 | 173M | 31.5G | ~3.0x  | [Google Drive](https://drive.google.com/file/d/19rvLU06bHVt6cmpE6thxSJen6e09lpd5/view?usp=sharing) |
|                 |      | 31.5G | ~2.0x  | [Google Drive](https://drive.google.com/file/d/1znnkmw3wBEXAoDhBq2kkrtNA5dsSzvpA/view?usp=sharing) |




## Fine-tuning ImageNet-22K pre-trained models
We give an example command for fine-tuning an ImageNet-22K pre-trained `CSWin-Base-224` model on ImageNet-1K:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=12345 main_buffer.py \
    --model CSWin_96_24322_base_224 --drop_path 0.2 --weight_decay 1e-8 \
    --batch_size 64 --lr 5e-5 --update_freq 1 \
    --warmup_epochs 0 --epochs 30 --end_epoch 30 \
    --cutmix 0 --mixup 0 --layer_decay 0.9 --input_size 224 \
    --use_amp true \
    --model_ema true --model_ema_eval true --model_ema_decay 0.9998 \
    --data_path /path/to/imagenet-1k \
    --output_dir /path/to/save/results \
    --finetune /path/to/checkpoint/ET_pp_in22k_pre_trained_speedup2x_cswinB.pth
```

- For other models, please change `--model`, `--finetune`, and `--input_size` accordingly. You can get the pre-trained models from the table above. 
- For better performance `--drop_path`, `--layer_decay`, and `--model_ema_decay` can be adjusted. In our paper, we determine these hyper-parameters on top of the baseline models, and directly use these obtained configurations for fine-tuning our ImageNet-22K pre-trained models.
