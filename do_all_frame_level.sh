collection=$1 
feature=$2
overwrite=0

# Augmentation for frame-level features
stride=$3

# Loss
loss=$4

# Generate augmented frame-level features and operate mean pooling to obtain video-level features in advance
python gene_aug_feat.py --collection $collection --feature $feature --stride $stride --overwrite $overwrite

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python train.py --collection $collection --feature $feature --stride $stride --loss $loss --overwrite $overwrite