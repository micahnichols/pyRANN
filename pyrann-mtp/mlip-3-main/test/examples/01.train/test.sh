#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
$MLP_EXE train 06.almtp train.cfg --save_to=$TMP_DIR/pot.almtp --iteration_limit=100 --al_mode=nbh
