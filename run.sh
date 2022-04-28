EPOCHS=600
export CUDA_VISIBLE_DEVICES=2,3
# python3 -O training_smaller.py \
#  --batch-size 2048 \
#  --lr 0.02 \
#  --epochs $EPOCHS \
#  --seed 11111 \
#  --decay 0 \
#  --workers 16 \
#  --optimizer Adam \
#  --duplicate 3 \
#  --train \
#  --sess tiny_imagenet_smaller

#  python3 -O training_smaller.py \
#  --batch-size 384 \
#  --lr 5e-3 \
#  --epochs 300 \
#  --seed 11111 \
#  --decay 0 \
#  --workers 16 \
#  --optimizer SGD \
#  --duplicate 10 \
#  --train \
#  --sess tiny_imagenet_smaller

## CIFAR-10
python3 -O training_smaller.py \
 --batch-size 2048 \
 --lr 0.02 \
 --epochs $EPOCHS \
 --seed 11111 \
 --decay 0 \
 --workers 16 \
 --optimizer Adam \
 --duplicate 10 \
 --train \
 --sess tiny_imagenet_smaller

export CUDA_VISIBLE_DEVICES=3
python3 -O training_smaller.py \
 --seed 11111 \
 --workers 16 \
 --epochs $EPOCHS \
 --sess tiny_imagenet_smaller \
 --resume
