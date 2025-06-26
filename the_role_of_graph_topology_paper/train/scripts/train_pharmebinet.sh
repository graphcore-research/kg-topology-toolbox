# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#!/bin/bash

# DistMult
python3 train.py --data ../datasets/data/pharmebinet --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 -s 16 --batch-size 48 --accum-factor 4 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --device-iter-inf 1 --half --loss-scale 256 --scoring-function DistMult --dim 300 --lr 0.003 --neg 8 --inference-device ipu --filter-test --return-topk --validation-split 0.0035 --test-split 0.0035 --test-on test

# RotatE
python3 train.py --data ../datasets/data/pharmebinet --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 -s 16 --batch-size 48 --accum-factor 4 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --device-iter-inf 1 --half --loss-scale 256 --scoring-function RotatE --dim 128 --lr 0.001 --neg 8 --scoring-norm 2 --inference-device ipu --filter-test --return-topk --validation-split 0.0035 --test-split 0.0035 --test-on test

# TransE
python3 train.py --data ../datasets/data/pharmebinet --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 -s 16 --batch-size 48 --accum-factor 4 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --device-iter-inf 1 --half --loss-scale 256 --scoring-function TransE --dim 256 --lr 0.00003 --neg 8 --scoring-norm 1 --inference-device ipu --filter-test --return-topk --validation-split 0.0035 --test-split 0.0035 --test-on test

# TripleRE
python3 train.py --data ../datasets/data/pharmebinet --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 -s 16 --batch-size 48 --accum-factor 4 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --device-iter-inf 1 --half --loss-scale 256 --scoring-function TripleRE --dim 256 --lr 0.0001 --neg 8 --scoring-norm 2 --inference-device ipu --filter-test --return-topk --validation-split 0.0035 --test-split 0.0035 --test-on test

# ConvE
python3 train.py --data ../datasets/data/pharmebinet --store-predictions --neg-adversarial-sampling --final-validation --epochs 50 --validation-epochs 0 -s 16 --batch-size 48 --accum-factor 4 --device-iter 8 --inference-batch-size 64 --inference-window-size 256 --device-iter-inf 1 --half --loss-scale 256 --scoring-function TripleRE --dim 256 --embedding-height 11 --lr 0.0001 --neg 4 --inference-device ipu --filter-test --return-topk --validation-split 0.0035 --test-split 0.0035 --test-on test
