#! /bin/bash
# check gpu info
nvidia-smi
# pytorch 04
PYTHON="$HOME/anaconda3/envs/tensorflow/bin/python"
##### experimental settings
# network config
ENCODER="resnet101"
DECODER="attention"
DATASET="nyu"
NUM_CLASSES=80
# datasets
# replace the DATA_DIR with your folder path to the dataset.
RGB_DIR="~/myDataset/NYU_v2/"
DEP_DIR="~/myDataset/NYU_v2/"
TRAIN_RGB_TXT="../datasets/nyu_path/train_rgb_12k.txt"
TRAIN_DEP_TXT="../datasets/nyu_path/train_depth_12k.txt" 
VAL_RGB_TXT="../datasets/nyu_path/valid_rgb.txt"
VAL_DEP_TXT="../datasets/nyu_path/valid_depth.txt"
# training settings
MODE="train"
GPU=True
EPOCHES=10
LR=2e-4
FINAL_LR=2e-3
WEIGHT_DECAY=5e-4
BATCHSIZE=6
BATCHSIZEVAL=6
EVAL_FREQ=1
THREADS=4
OPTIMIZER="sgd"
SCHEDULER="poly"
POWER=0.9
USE_WEIGHTS=False
CLASSIFIER="OR"
INFERENCE="soft"
EPS=0.0
PRIOR="gaussian"
OHEMTHRES=0.7
OHEMKEEP=100000
ALPHA=0
BETA=0
# set the output path of checkpoints, training log.
WORKSPACE_DIR="../workspace/"
LOG_DIR="log_${ENCODER}${DECODER}_${DATASET}_${CLASSIFIER}_190703a"
#RESUME="${WORKSPACE_DIR}log_${ENCODER}${DECODER}_${DATASET}_${CLASSIFIER}_190703a/best.pkl"
########################################################################################################################
#  Training
########################################################################################################################
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE \
                            --classes $NUM_CLASSES --epochs $EPOCHES --eval-freq $EVAL_FREQ --threads $THREADS \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --train-rgb $TRAIN_RGB_TXT --train-dep $TRAIN_DEP_TXT \
                            --val-rgb $VAL_RGB_TXT --val-dep $VAL_DEP_TXT --batch  $BATCHSIZE --batch-val $BATCHSIZEVAL \
                            --optimizer $OPTIMIZER --weight-decay $WEIGHT_DECAY --lr $LR --final-lr $FINAL_LR --gpu $GPU \
                            --scheduler $SCHEDULER --power $POWER --random-flip --random-jitter --random-crop \
                            --workdir $WORKSPACE_DIR --logdir $LOG_DIR --pretrain --eps $EPS --prior $PRIOR --use-weights $USE_WEIGHTS \
                            --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --alpha $ALPHA --beta $BETA --resume $RESUME &&

########################################################################################################################
#  Testing
########################################################################################################################
# dataset
TEST_RGB_TXT="../datasets/nyu_path/valid_rgb.txt"
TEST_DEP_TXT="../datasets/nyu_path/valid_depth.txt"
TEST_RES_DIR="res"
# testing settings
MODE="test"
GPU=True
TEST_USE_FLIP=True
TEST_USE_MS=True
INFERENCE='soft'
TEST_CHECKPOINT="best.pkl"
TEST_RESTORE_FROM="${WORKSPACE_DIR}${LOG_DIR}/${TEST_CHECKPOINT}"
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE --classes $NUM_CLASSES \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --test-rgb $TEST_RGB_TXT --test-dep $TEST_DEP_TXT \
                            --gpu $GPU --use-flip $TEST_USE_FLIP --use-ms $TEST_USE_MS --logdir $LOG_DIR --resdir $TEST_RES_DIR  \
                            --resume $TEST_RESTORE_FROM 

