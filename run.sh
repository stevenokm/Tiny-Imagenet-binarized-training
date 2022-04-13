export CUDA_VISIBLE_DEVICES=2,3
# python3 -O training_smaller.py \
#  --batch-size 384 \
#  --lr 0.02 \
#  --epochs 300 \
#  --seed 11111 \
#  --decay 0 \
#  --workers 16 \
#  --optimizer Adam \
#  --train \
#  --sess tiny_imagenet_smaller
 python3 -O training_smaller.py \
 --batch-size 384 \
 --lr 0.001 \
 --epochs 300 \
 --seed 11111 \
 --decay 0 \
 --workers 16 \
 --optimizer SGD \
 --train \
 --sess tiny_imagenet_smaller

export CUDA_VISIBLE_DEVICES=0
python3 -O training_smaller.py \
 --seed 11111 \
 --workers 16 \
 --epochs 301 \
 --sess tiny_imagenet_smaller \
 --resume
