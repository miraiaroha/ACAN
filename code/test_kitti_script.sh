#! /bin/bash
PYTHON="$HOME/anaconda3/envs/tensorflow/bin/python"
# network config
ENCODER="resnet101"
DECODER="attention"
DATASET="kitti"
## experimental settings
CLASSIFIER="OR"
INFERENCE="soft"
NUM_CLASSES=80
# dataset
RGB_DIR="~/myDataset/KITTI/raw_data_KITTI/"
DEP_DIR="~/myDataset/KITTI/datasets_KITTI/"
TEST_RGB_TXT="../datasets/kitti_path/eigen_test_files.txt"
TEST_DEP_TXT="../datasets/kitti_path/eigen_test_depth_files.txt"
TEST_RES_DIR="res"
# testing settings
MODE="test"
GPU=True
TEST_USE_FLIP=True
TEST_USE_MS=False
INFERENCE='soft'
# set the output path of checkpoints, training log.
WORKSPACE_DIR="../workspace/"
LOG_DIR="log_${ENCODER}${DECODER}_${DATASET}_${CLASSIFIER}_200112att"
TEST_CHECKPOINT="best.pkl"
TEST_RESTORE_FROM="${WORKSPACE_DIR}${LOG_DIR}/${TEST_CHECKPOINT}"
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE --classes $NUM_CLASSES \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --test-rgb $TEST_RGB_TXT --test-dep $TEST_DEP_TXT \
                            --gpu True --use-flip $TEST_USE_FLIP --use-ms $TEST_USE_MS --logdir $LOG_DIR --resdir $TEST_RES_DIR  \
                            --resume $TEST_RESTORE_FROM 