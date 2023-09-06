#!/usr/bin/env bash

export SEQ_LENGTH=128
export HS=1024
export TP=1
export PP=1
export N_LAYERS=8
export N_AH=8
export ACT_CHKPNT_GRANULARITY=full

./test.sh
# bash ./test_lightning.sh
