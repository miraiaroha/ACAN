#! /bin/bash
# check gpu info
nvidia-smi
# pytorch 04
PYTHON="$HOME/anaconda3/envs/tensorflow/bin/python"
# network config
ENCODER="resnet50"
DECODER="attention"
DATASET="kitti"
NUM_CLASSES=80
# datasets
# replace the DATA_DIR with your folder path to the dataset.
RGB_DIR="~/myDataset/KITTI/raw_data_KITTI/"
DEP_DIR="~/myDataset/KITTI/datasets_KITTI/"
TRAIN_RGB_TXT="../datasets/kitti_path/eigen_train_files.txt"
TRAIN_DEP_TXT="../datasets/kitti_path/eigen_train_depth_files.txt"
VAL_RGB_TXT="../datasets/kitti_path/eigen_test_files.txt"
VAL_DEP_TXT="../datasets/kitti_path/eigen_test_depth_files.txt"
# training settings
MODE="train"
GPU=True
EPOCHES=50
LR=2e-4
FINAL_LR=2e-3
WEIGHT_DECAY=5e-4
BATCHSIZE=8
BATCHSIZEVAL=8
EVAL_FREQ=1
THREADS=4
OPTIMIZER="sgd"
SCHEDULER="poly"
POWER=0.9
USE_WEIGHTS=False
CLASSIFIER="CE"
INFERENCE="soft"
EPS=0.0
PRIOR="uniform"
OHEMTHRES=0.7
OHEMKEEP=100000
ALPHA=0
BETA=0
# set the output path of checkpoints, training log.
WORKSPACE_DIR="../workspace/"
TRAIN_LOG_DIR="log_${ENCODER}${DECODER}_${DATASET}_${CLASSIFIER}"
########################################################################################################################
#  Training
########################################################################################################################
## experimental settings
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE \
                            --classes $NUM_CLASSES --epochs $EPOCHES --eval-freq $EVAL_FREQ --threads $THREADS \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --train-rgb $TRAIN_RGB_TXT --train-dep $TRAIN_DEP_TXT \
                            --val-rgb $VAL_RGB_TXT --val-dep $VAL_DEP_TXT --batch  $BATCHSIZE --batch-val $BATCHSIZEVAL \
                            --optimizer $OPTIMIZER --weight-decay $WEIGHT_DECAY --lr $LR --final-lr $FINAL_LR --gpu $GPU \
                            --scheduler $SCHEDULER --power $POWER --random-flip --random-scale --random-rotate --random-jitter --random-crop \
                            --workdir $WORKSPACE_DIR --logdir $LOG_DIR --pretrain --eps $EPS --prior $PRIOR --use-weights $USE_WEIGHTS \
                            --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --alpha $ALPHA --beta $BETA &&

########################################################################################################################
#  Testing
########################################################################################################################
# dataset
TEST_RGB_TXT="../datasets/kitti_path/eigen_test_files.txt"
TEST_DEP_TXT="../datasets/kitti_path/eigen_test_depth_files.txt"
TEST_RES_DIR="res"
# testing settings
MODE="test"
GPU=True
TEST_USE_FLIP=True
TEST_USE_MS=True
INFERENCE='soft'
TEST_CHECKPOINT="best.pkl"
TEST_RESTORE_FROM="${WORKSPACE_DIR}${TRAIN_LOG_DIR}/${TEST_CHECKPOINT}"
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE --classes $NUM_CLASSES \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --test-rgb $TEST_RGB_TXT --test-dep $TEST_DEP_TXT \
                            --gpu $GPU --use-flip $TEST_USE_FLIP --use-ms $TEST_USE_MS --logdir $LOG_DIR --resdir $TEST_RES_DIR  \
                            --resume $TEST_RESTORE_FROM 

                        

