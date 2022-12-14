import os
import argparse

parser = argparse.ArgumentParser(description='EfficientTrain')
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_dir', default='output/EfficientTrain', type=str)

parser.add_argument('--model', type=str)

parser.add_argument('--final_bs', default=256, type=int)
parser.add_argument('--num_gpus', default=8, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--num_workers', default=10, type=int)

args = parser.parse_args()

# This repo currently support training the following models on ImageNet-1K.
# The drop path rate follows from the original papers.
default_dp_dict = {
    'resnet50': 0.0,
    'convnext_tiny': 0.1,
    'convnext_small': 0.4,
    'convnext_base': 0.5,
    'deit_tiny_patch16_224': 0.1,
    'deit_small_patch16_224': 0.1,
    'swin_tiny': 0.2,
    'swin_small': 0.3,
    'swin_base': 0.5,
    'CSWin_64_12211_tiny_224': 0.2,
    'CSWin_64_24322_small_224': 0.4,
    'CSWin_96_24322_base_224': 0.5,
}

# These configs follow from the official implementations of the models.
additional_configs = {
    'resnet50': '',
    'convnext_tiny': ' --model_ema true --model_ema_eval true ',
    'convnext_small': ' --model_ema true --model_ema_eval true ',
    'convnext_base': ' --model_ema true --model_ema_eval true ',
    'deit_tiny_patch16_224': ' --use_amp true --clip_grad 5.0 ',
    'deit_small_patch16_224': ' --use_amp true --clip_grad 5.0 ',
    'swin_tiny': ' --use_amp true --clip_grad 5.0 ',
    'swin_small': ' --use_amp true --clip_grad 5.0 ',
    'swin_base': ' --use_amp true --clip_grad 5.0 ',
    'CSWin_64_12211_tiny_224': ' --use_amp true --model_ema true --model_ema_eval true ',
    'CSWin_64_24322_small_224': ' --use_amp true --model_ema true --model_ema_eval true ',
    'CSWin_96_24322_base_224': ' --use_amp true --model_ema true --model_ema_eval true ',
}

b_list = [160,] * 3 + [192,] * 1 + [224,] * 1
print('EfficientTrain_b_list:', b_list)

for ET_index in range(5):

    current_bs = int(args.final_bs * 2 if ET_index < 4 else args.final_bs)
    current_update_freq = int(4096 / args.num_gpus / current_bs)
    command = f" \
        python -m torch.distributed.launch --nproc_per_node={args.num_gpus} main.py \
        --model {args.model} --drop_path {default_dp_dict[args.model]} \
        --batch_size {current_bs} --lr 4e-3 --update_freq {current_update_freq} \
        --data_path {args.data_path} --output_dir ./{args.output_dir} --num_workers {args.num_workers} \
        --epochs {args.epochs} --warmup_epochs 20 --end_epoch {int(args.epochs / 5 * (ET_index + 1))} \
        --input_size {b_list[ET_index]} --aa rand-m{int(2 * ET_index + 1)}-mstd0.5-inc1 \
    " + additional_configs[args.model]

    print(command)
    killed = os.system(command)

    if killed:
        current_bs = int(args.final_bs * 2 if ET_index < 4 else args.final_bs)
        current_bs = int(current_bs / 2)
        current_update_freq = int(4096 / args.num_gpus / current_bs)
        command = f" \
            python -m torch.distributed.launch --nproc_per_node={args.num_gpus} main.py \
            --model {args.model} --drop_path {default_dp_dict[args.model]} \
            --batch_size {current_bs} --lr 4e-3 --update_freq {current_update_freq} \
            --data_path {args.data_path} --output_dir ./{args.output_dir} --num_workers {args.num_workers} \
            --epochs {args.epochs} --warmup_epochs 20 --end_epoch {int(args.epochs / 5 * (ET_index + 1))} \
            --input_size {b_list[ET_index]} --aa rand-m{int(2 * ET_index + 1)}-mstd0.5-inc1 \
        " + additional_configs[args.model]

        print(command)
        os.system(command)