# linear_probe.sh
#!/bin/bash


python main_lincls.py \
  -a resnet50 \
  --lr 0.0125 \
  --epochs 200 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:58245' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dataset web-aircraft \
  --save-dir ./linear_probe_results \
  --save-freq 25 \
  --wd 0.001 \
  --pretrained ./moco_strong/checkpoint_0199.pth.tar \
  