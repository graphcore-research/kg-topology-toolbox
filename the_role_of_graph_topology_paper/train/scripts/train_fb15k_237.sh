# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#!/bin/bash

# DistMult
python3 train.py --data ../datasets/data/fb15k-237 --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 --batch-size 128 --inference-batch-size 128 --inference-window-size 256 --half --loss-scale 256 --scoring-function DistMult --dim 4096 --lr 0.001 --neg 8 --inference-device ipu --filter-test --return-topk --test-on test 

# RotatE
python3 train.py --data ../datasets/data/fb15k-237 --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 --batch-size 128 --inference-batch-size 128 --inference-window-size 256 --half --loss-scale 256 --scoring-function RotatE --dim 1024 --lr 0.003 --neg 8 --scoring-norm 2 --inference-device ipu --filter-test --return-topk --test-on test 

# TransE
python3 train.py --data ../datasets/data/fb15k-237 --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 --batch-size 128 --inference-batch-size 128 --inference-window-size 256 --half --loss-scale 256 --scoring-function TransE --dim 2048 --lr 0.0001 --neg 8 --scoring-norm 1 --inference-device ipu --filter-test --return-topk --test-on test 

# TripleRE
python3 train.py --data ../datasets/data/fb15k-237 --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 --batch-size 128 --inference-batch-size 128 --inference-window-size 256 --half --loss-scale 256 --scoring-function TripleRE --dim 256 --lr 0.001 --neg 8 --scoring-norm 1 --inference-device ipu --filter-test --return-topk --test-on test 

# ConvE
python3 train.py --data ../datasets/data/fb15k-237 --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 --batch-size 64 --accum-factor 12 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --half --loss-scale 256 --scoring-function ConvE --dim 676 --embedding-height 13 --lr 0.001 --neg 4 --inference-device ipu --filter-test --return-topk --test-on test 
