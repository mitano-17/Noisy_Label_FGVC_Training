# train_vanilla.sh
#!/bin/bash

# From scratch (no pretrained weights)

python main_super.py \
  --dataset web-aircraft \
  --epochs 1 \
  --batch-size 32 \
  --lr 0.0125 \
  --save-freq 25 \
  --aug-strength moderate \
  --moco-path moco_mod/checkpoint_0399.pth.tar \
  --save-dir fine_super_mod_aug/ \