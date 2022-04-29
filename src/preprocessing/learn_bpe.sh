#!/bin/bash
DATA_DIR=$1
OUT_DIR=$2
FAST=~/fastBPE/fast

# Learn codes
$FAST learnbpe 50000 ${OUT_DIR}/train.source ${OUT_DIR}/train.target > ${DATA_DIR}/codes

# Apply codes to train
SPLIT=train
for LANG in source target
  do
    $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} ${DATA_DIR}/codes
  done

# Get train vocabulary
$FAST getvocab ${OUT_DIR}/train.source.bpe ${OUT_DIR}/train.target.bpe  > ${DATA_DIR}/vocab.bpe

# Apply codes to valid and test
for SPLIT in valid test
do
  for LANG in source target
  do
    $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} ${DATA_DIR}/codes
  done
done
