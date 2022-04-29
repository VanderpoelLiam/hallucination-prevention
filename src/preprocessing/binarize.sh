#!/bin/bash

OUT_DIR=$1
DATA_DIR=$2

fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "${OUT_DIR}/train.bpe" \
    --validpref "${OUT_DIR}/valid.bpe" \
    --testpref "${OUT_DIR}/test.bpe" \
    --destdir "${OUT_DIR}" \
    --srcdict "$DATA_DIR/dict.txt" \
    --tgtdict "$DATA_DIR/dict.txt" \
    --workers 60
