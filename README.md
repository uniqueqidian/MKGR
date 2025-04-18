## dissertation title：

Research and Implementation of Multi-modal Knowledge Graph Reasoning Based on Large Models

## Dependencies

```bash
pip install -r requirements.txt
```

#### Details

- Python
- numpy
- scikit_learn
- torch
- tqdm
  Our code is based on OpenKE, an open-source KGC project. You can refer to the [OpenKE repo](https://github.com/thunlp/OpenKE) to build the environment.

## Train and Evaluation

You can refer to the training scripts in `scripts/` to reproduce our experiment results. Here is an example for DB15K dataset.

```bash
DATA=DB15K
EMB_DIM=250
NUM_BATCH=1024
MARGIN=12
LR=1e-4
LRG=1e-4
NEG_NUM=128
MU=0.0001
EPOCH=1000

CUDA_VISIBLE_DEVICES=1 nohup python run_adv_wgan_gp_3modal.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=$EPOCH \
  -dim=$EMB_DIM \
  -adv_num=$ADV \
  -save=$DATA-$NUM_BATCH-$EMB_DIM-$NEG_NUM-$MU-$MARGIN-$LR-$EPOCH \
  -neg_num=$NEG_NUM \
  -mu=$MU \
  -learning_rate=$LR \
  -lrg=$LRG >$DATA-$EMB_DIM-$NUM_BATCH-$NEG_NUM-$MU-$MARGIN-$LR-$EPOCH.txt &
```
