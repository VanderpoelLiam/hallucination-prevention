#!/bin/sh
WORK_DIR=$1
FILENAME=$2
SAVE_FILE=$3

echo "Extracting target and hypothesis summaries ..."
grep ^T- $FILENAME | cut -f2- > $WORK_DIR/target.raw
grep ^H- $FILENAME | cut -f3- > $WORK_DIR/hypothesis.raw

echo "Removing BPE ..."
for l in target hypothesis; do
  sed 's/@@ //g' $WORK_DIR/${l}.raw > $WORK_DIR/${l}.debpe
  rm $WORK_DIR/${l}.raw
done

SCRIPTS=~/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl

echo "Removing tokenization ..."
for l in target hypothesis; do
  cat $WORK_DIR/${l}.debpe |
  perl $DETOKENIZER -threads 8 -q -l en > $WORK_DIR/${l}.detok
  rm $WORK_DIR/${l}.debpe
done

echo "Running files2rouge ..."
files2rouge $WORK_DIR/target.detok $WORK_DIR/hypothesis.detok > $WORK_DIR/$SAVE_FILE

echo "ROUGE scores: "
python src/extract_score.py $WORK_DIR/$SAVE_FILE
