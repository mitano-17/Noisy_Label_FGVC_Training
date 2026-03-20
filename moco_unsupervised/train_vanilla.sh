# train_vanilla.sh
#!/bin/bash


python main_moco.py \
  -a resnet50 \
  --lr 0.0125 \
  --epochs 200 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:58245' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dataset web-aircraft \
  --aug-strength strong \
  --mlp \
  --moco-t 0.2 \
  --cos \
  --moco-k 8192 \
  --save-dir ./moco_strong \
  --save-freq 25 \
  --wd 0.001 \
  