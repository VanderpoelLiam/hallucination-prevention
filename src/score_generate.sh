#!/bin/sh
NAME=generate_$1

EXPERIMENT_DIR=logs/experiments
ROUGE_DIR=logs/rouge
WORK_DIR=$ROUGE_DIR/$NAME

FILENAME=$EXPERIMENT_DIR/${NAME}.log

mkdir $WORK_DIR -p

src/compute_rouge.sh $WORK_DIR $FILENAME score
