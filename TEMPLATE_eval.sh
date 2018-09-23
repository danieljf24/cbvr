rootpath=@@@rootpath@@@
collection=@@@collection@@@
overwrite=@@@overwrite@@@

checkpoint_path=@@@model_path@@@/model_best.pth.tar

gpu=1
for test_set in val test
do
CUDA_VISIBLE_DEVICES=$gpu python test.py --rootpath $rootpath --collection $collection --checkpoint_path $checkpoint_path --test_set $test_set --overwrite $overwrite
done
