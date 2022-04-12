export CUDA_VISIBLE_DEVICES=0,1
python3 -O training_smaller.py \
 --batch-size 128 \
 --lr 0.02 \
 --epochs 200 \
 --seed 11111 \
 --decay 0 \
 --sess tiny_imagenet_smaller
 
