for collection in tgif-msrvtt10k-msvd tv2016test tv2016train tv2017test tv2018test
do
for feat in pyresnet-152_imagenet11k,flatten0_output,os pyresnext-101_rbps13k,flatten0_output,os
do
python norm_feat.py /home/daniel/VisualSearch/trecvid2018/${collection}/FeatureData/$feat
done
done
