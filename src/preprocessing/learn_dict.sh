#!/bin/bash

OUT_DIR=$1
DATA_DIR=$2

for SPLIT in train test valid
do
  cat "$OUT_DIR/$SPLIT.bpe.target" "$OUT_DIR/$SPLIT.bpe.source" > "$OUT_DIR/$SPLIT.bpe.full"
done

fairseq-preprocess \
    --source-lang "full" \
    --trainpref "${OUT_DIR}/train.bpe" \
    --destdir "${OUT_DIR}" \
    --only-source \
    --dict-only \
    --workers 60

mv ${OUT_DIR}/dict.full.txt  $DATA_DIR/dict.txt
