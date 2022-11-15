#!/bin/sh

NMT_ROOT=~/nmt/

src=zh
tgt=en

SCRIPTS=${NMT_ROOT}/tools/mosesdecoder/scripts
BPEROOT=${NMT_ROOT}/tools/subword-nmt

TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
DETOKENIZER=${SCRIPTS}/tokenizer/detokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
TRAIN_TC=${SCRIPTS}/recaser/train-truecaser.perl
TC=${SCRIPTS}/recaser/truecase.perl
DETC=${SCRIPTS}/recaser/detruecase.perl
NORM_PUNC=${SCRIPTS}/tokenizer/normalize-punctuation.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
MULTI_BLEU=${SCRIPTS}/generic/multi-bleu.perl
MTEVAL_V14=${SCRIPTS}/generic/mteval-v14.pl

data_dir=${NMT_ROOT}/dataset/ncv15/
model_dir=${data_dir}/models
utils=${NMT_ROOT}/lite-transformer/rk_tools
