import os
import argparse


parser = argparse.ArgumentParser(description='EfficientTrain')
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--seed', default='', type=str)
parser.add_argument('--epoch', default=200, type=int)


args = parser.parse_args()

res_list = [96,] * 2 + [160,] * 4 + [224,] * 4
print('res_list:', res_list)

bs_list = [512,] * 2 + [512,] * 4 + [256,] * 4
up_freq_list = [1,] * 2 + [1,] * 4 + [2,] * 4
print('bs_list:', bs_list)
print('up_freq_list:', up_freq_list)

replay_times_list = [2,] * 2 + [1,] * 4 + [1,] * 4
replay_buffer_size_list = [1024,] * 2 + [0,] * 4 + [0,] * 4
print('replay_times_list:', replay_times_list)
print('replay_buffer_size_list:', replay_buffer_size_list)



epoch_scale_ratio = {
    64:  (224 * 224) / (64 * 64),
    96:  (224 * 224) / (96 * 96),
    128: (224 * 224) / (128 * 128),
    160: (224 * 224) / (160 * 160),
    192: (224 * 224) / (192 * 192),
    224: 1,
}

for ET_index in range(10):
    
    tag = args.tag
    print()
    print('save at: ', tag)
    print()

    rp_epoch = int(
        args.epoch * epoch_scale_ratio[res_list[ET_index]] / replay_times_list[ET_index]
        )
    rp_warmup_epoch = int(
        20 / replay_times_list[ET_index]
        )

    command = f" \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=11333 main_buffer.py \
        --data_path /home/data/imagenet/  --num_workers 10 \
        --output_dir ./{tag} \
        --epochs {rp_epoch} --end_epoch {int(rp_epoch / 10 * (ET_index + 1))} \
        --warmup_epochs {rp_warmup_epoch} \
        --aa rand-m{ET_index}-mstd0.5-inc1 \
        --input_size {res_list[ET_index]} \
        --model swin_tiny --drop_path {0.2 * args.epoch / 200} \
        --use_amp true --clip_grad 5.0 \
        --batch_size {int(bs_list[ET_index] / replay_times_list[ET_index])} --lr 4e-3 --update_freq {up_freq_list[ET_index]} \
        --replay_times {replay_times_list[ET_index]} --replay_buffer_size {replay_buffer_size_list[ET_index]} \
        --seed {args.seed} \
    "
    print()
    print(command)
    print()
    os.system(command)