collection=$1 
feature=$2
overwrite=0

# Augmentation for video-level features
aug_prob=0.5

# Loss
loss=$3

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python train.py --collection $collection --feature $feature --aug_prob $aug_prob --loss $loss --overwrite $overwrite