# EPOCHS=600
# SESS=tiny_imagenet_smaller
# SEED=11111

# # tiny-imagenet Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#     --batch-size 2048 \
#     --lr 0.02 \
#     --epochs $EPOCHS \
#     --seed $SEED \
#     --decay 0 \
#     --workers 16 \
#     --optimizer Adam \
#     --duplicate 3 \
#     --train \
#     --sess $SESS

# # tiny-imagenet Test
# export CUDA_VISIBLE_DEVICES=0
# python3 -O training_smaller.py \
#     --batch-size 384 \
#     --lr 5e-3 \
#     --epochs 300 \
#     --seed $SEED \
#     --decay 0 \
#     --workers 16 \
#     --optimizer SGD \
#     --duplicate 10 \
#     --train \
#     --sess $SESS

EPOCHS=500
SESS=cifar10_smaller
SEED=11111
NOISE=0.0

# # CIFAR-10 Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#     --batch-size 2048 \
#     --lr 0.02 \
#     --epochs $EPOCHS \
#     --seed $SEED \
#     --decay 0 \
#     --workers 16 \
#     --optimizer Adam \
#     --duplicate 10 \
#     --train \
#     --sess $SESS

# CIFAR-10 Test
export CUDA_VISIBLE_DEVICES=1
python3 -O training_smaller.py \
    --seed $SEED \
    --workers 16 \
    --epochs $EPOCHS \
    --sess $SESS \
    --mem_fault baseline \
    --noise $NOISE \
    --resume

# CIFAR-10 sub-project III

sub_proj_3() {

    FAULTY_BIT=$1
    FAULTY_BIT=${FAULTY_BIT:=0}
    NEIGHBOR_OFFSET=$2
    NEIGHBOR_OFFSET=${NEIGHBOR_OFFSET:=-1}
    WSCONV=$3
    WSCONV=${WSCONV:="false"}

    if [ $WSCONV == "true" ]; then
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault faulty \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --wsconv \
            --resume
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault reparied_n \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --wsconv \
            --resume
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault reparied_s \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --wsconv \
            --resume
    else
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault faulty \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --resume
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault reparied_n \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --resume
        python3 -O training_smaller.py \
            --seed $SEED \
            --workers 16 \
            --epochs $EPOCHS \
            --sess $SESS \
            --mem_fault reparied_s \
            --faulty_bit $FAULTY_BIT \
            --neighbor_offset $NEIGHBOR_OFFSET \
            --resume
    fi
}

sub_proj_3 6 -1 "false"
sub_proj_3 6 -2 "false"
sub_proj_3 6 -3 "false"
sub_proj_3 6 -4 "false"
sub_proj_3 6 -5 "false"
sub_proj_3 6 -6 "false"

sub_proj_3 5 1 "false"
sub_proj_3 5 -1 "false"
sub_proj_3 5 -2 "false"
sub_proj_3 5 -3 "false"
sub_proj_3 5 -4 "false"
sub_proj_3 5 -5 "false"

sub_proj_3 4 2 "false"
sub_proj_3 4 1 "false"
sub_proj_3 4 -1 "false"
sub_proj_3 4 -2 "false"
sub_proj_3 4 -3 "false"
sub_proj_3 4 -4 "false"

sub_proj_3 3 3 "false"
sub_proj_3 3 2 "false"
sub_proj_3 3 1 "false"
sub_proj_3 3 -1 "false"
sub_proj_3 3 -2 "false"
sub_proj_3 3 -3 "false"

sub_proj_3 2 4 "false"
sub_proj_3 2 3 "false"
sub_proj_3 2 2 "false"
sub_proj_3 2 1 "false"
sub_proj_3 2 -1 "false"
sub_proj_3 2 -2 "false"

# wsconv

EPOCHS=600
SESS=cifar10_smaller_wsconv
SEED=11111

# # CIFAR-10 Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#     --batch-size 2048 \
#     --lr 0.02 \
#     --epochs $EPOCHS \
#     --seed $SEED \
#     --decay 0 \
#     --workers 16 \
#     --optimizer Adam \
#     --duplicate 10 \
#     --train \
#     --noise $NOISE \
#     --wsconv \
#     --sess $SESS

# CIFAR-10 Test
export CUDA_VISIBLE_DEVICES=1
python3 -O training_smaller.py \
    --seed $SEED \
    --workers 16 \
    --epochs $EPOCHS \
    --sess $SESS \
    --mem_fault baseline \
    --noise $NOISE \
    --wsconv \
    --resume

# CIFAR-10 sub-project II & III

sub_proj_3 6 -1 "true"
sub_proj_3 6 -2 "true"
sub_proj_3 6 -3 "true"
sub_proj_3 6 -4 "true"
sub_proj_3 6 -5 "true"
sub_proj_3 6 -6 "true"

sub_proj_3 5 1 "true"
sub_proj_3 5 -1 "true"
sub_proj_3 5 -2 "true"
sub_proj_3 5 -3 "true"
sub_proj_3 5 -4 "true"
sub_proj_3 5 -5 "true"

sub_proj_3 4 2 "true"
sub_proj_3 4 1 "true"
sub_proj_3 4 -1 "true"
sub_proj_3 4 -2 "true"
sub_proj_3 4 -3 "true"
sub_proj_3 4 -4 "true"

sub_proj_3 3 3 "true"
sub_proj_3 3 2 "true"
sub_proj_3 3 1 "true"
sub_proj_3 3 -1 "true"
sub_proj_3 3 -2 "true"
sub_proj_3 3 -3 "true"

sub_proj_3 2 4 "true"
sub_proj_3 2 3 "true"
sub_proj_3 2 2 "true"
sub_proj_3 2 1 "true"
sub_proj_3 2 -1 "true"
sub_proj_3 2 -2 "true"
