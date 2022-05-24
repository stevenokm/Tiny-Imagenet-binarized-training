# EPOCHS=600
# SESS=tiny_imagenet_smaller
# SEED=11111

# # tiny-imagenet Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#  --batch-size 2048 \
#  --lr 0.02 \
#  --epochs $EPOCHS \
#  --seed $SEED \
#  --decay 0 \
#  --workers 16 \
#  --optimizer Adam \
#  --duplicate 3 \
#  --train \
#  --sess $SESS

# # tiny-imagenet Test
# export CUDA_VISIBLE_DEVICES=0
# python3 -O training_smaller.py \
#  --batch-size 384 \
#  --lr 5e-3 \
#  --epochs 300 \
#  --seed $SEED \
#  --decay 0 \
#  --workers 16 \
#  --optimizer SGD \
#  --duplicate 10 \
#  --train \
#  --sess $SESS

EPOCHS=500
SESS=cifar10_smaller
SEED=11111
NOISE=1.0

# # CIFAR-10 Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#  --batch-size 2048 \
#  --lr 0.02 \
#  --epochs $EPOCHS \
#  --seed $SEED \
#  --decay 0 \
#  --workers 16 \
#  --optimizer Adam \
#  --duplicate 10 \
#  --train \
#  --sess $SESS

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

# # CIFAR-10 sub-project III
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault faulty \
#  --resume
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault reparied_n \
#  --resume
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault reparied_s \
#  --resume

# wsconv

EPOCHS=600
SESS=cifar10_smaller_wsconv
SEED=11111

# # CIFAR-10 Train
# export CUDA_VISIBLE_DEVICES=0,1
# python3 -O training_smaller.py \
#  --batch-size 2048 \
#  --lr 0.02 \
#  --epochs $EPOCHS \
#  --seed $SEED \
#  --decay 0 \
#  --workers 16 \
#  --optimizer Adam \
#  --duplicate 10 \
#  --train \
#  --noise $NOISE \
#  --wsconv \
#  --sess $SESS

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

# # CIFAR-10 sub-project III
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault faulty \
#  --wsconv \
#  --resume
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault reparied_n \
#  --wsconv \
#  --resume
# python3 -O training_smaller.py \
#  --seed $SEED \
#  --workers 16 \
#  --epochs $EPOCHS \
#  --sess $SESS \
#  --mem_fault reparied_s \
#  --wsconv \
#  --resume
