#!/bin/sh
SCORE_REFERENCE='false'

while getopts ':s' 'OPTKEY'; do
    case ${OPTKEY} in
        's')
            SCORE_REFERENCE='true'
            ;;
    esac
done

shift $((OPTIND-1))

SUM_MODEL=$1
LANG_MODEL=$2
LAMBDA=$3

if [ "$SCORE_REFERENCE" = true ]
then
  fairseq-generate \
  data/xsum-hallucination \
  --path checkpoints/summarization_model/$SUM_MODEL/checkpoint_best.pt \
  --batch-size 16 \
  --beam 5 \
  --truncate-source \
  --score-reference \
  --lm-path checkpoints/$LANG_MODEL/checkpoint_best.pt \
  --lm-weight -$LAMBDA
else
  fairseq-generate \
  data/xsum-hallucination \
  --path checkpoints/summarization_model/$SUM_MODEL/checkpoint_best.pt \
  --batch-size 16 \
  --beam 5 \
  --truncate-source \
  --lm-path checkpoints/$LANG_MODEL/checkpoint_best.pt \
  --lm-weight -$LAMBDA
fi
