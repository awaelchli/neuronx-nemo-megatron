#!/usr/bin/env bash

export SEQ_LENGTH=2048
export HS=4096
export TP=8
export PP=1
export N_LAYERS=32
export N_AH=32
export ACT_CHKPNT_GRANULARITY=full

# ./test.sh
./test_lightning.sh

# cd neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
# bash gpt_6.7b.sh