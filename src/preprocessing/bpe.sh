#!/bin/bash
DATA_DIR=$1
OUT_DIR=$2
FAST=~/fastBPE/fast

# Apply codes to train, valid and test
for SPLIT in train valid test
do
  for LANG in source target
  do
    $FAST applybpe ${OUT_DIR}/${SPLIT}.bpe.${LANG} ${OUT_DIR}/${SPLIT}.${LANG} ${DATA_DIR}/codes
  done
done
