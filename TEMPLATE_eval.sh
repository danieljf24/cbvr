rootpath=@@@rootpath@@@
collection=@@@collection@@@
overwrite=@@@overwrite@@@

checkpoint_path=@@@model_path@@@/model_best.pth.tar

gpu=0
for test_set in val
do
  for strategy in 1 2
  do
    CUDA_VISIBLE_DEVICES=$gpu python test.py --rootpath $rootpath --collection $collection --checkpoint_path $checkpoint_path --test_set $test_set --overwrite $overwrite --strategy $strategy
  done
done
