#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

#echo $(pwd TMP_DIR)

# Body:
# This converts OUTCAR to the internal format out/relax.cfg.
$MLP_EXE convert --input_format=outcar OUTCAR out/relax.cfg
