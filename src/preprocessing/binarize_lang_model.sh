#!/bin/bash

OUT_DIR=$1
DATA_DIR=$2
LANG_DIR=$3

for SPLIT in train test valid
do
  cat "$OUT_DIR/$SPLIT.bpe.target" "$OUT_DIR/$SPLIT.bpe.source" > "$OUT_DIR/$SPLIT.bpe.full"
done

fairseq-preprocess \
    --only-source \
    --trainpref "$OUT_DIR/train.bpe.target" \
    --validpref "$OUT_DIR/valid.bpe.target" \
    --testpref "$OUT_DIR/test.bpe.target" \
    --destdir "$LANG_DIR" \
    --workers 60 \
    --srcdict $DATA_DIR/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "$OUT_DIR/train.bpe.full" \
    --validpref "$OUT_DIR/valid.bpe.full" \
    --testpref "$OUT_DIR/test.bpe.full" \
    --destdir "$LANG_DIR-full" \
    --workers 60 \
    --srcdict $DATA_DIR/dict.txt

cp $DATA_DIR/dict.txt $OUT_DIR/dict.txt
