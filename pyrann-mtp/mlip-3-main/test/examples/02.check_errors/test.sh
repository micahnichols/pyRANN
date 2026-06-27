#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
$MLP_EXE check_errors pot.almtp train.cfg --log=stdout --report_to=$TMP_DIR/errors.txt
