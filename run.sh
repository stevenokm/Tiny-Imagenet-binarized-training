export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -O training_smaller.py \
 --batch-size 128 \
 --lr 0.1 \
 --epochs 50 \
 --seed 11111 \
 --sess tiny_imagenet_smaller
 
