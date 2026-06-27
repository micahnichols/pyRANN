#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
$MLP_EXE cut_extrapolative_nbh selected.cfg $TMP_DIR/spherical.cfg --cutoff=8 
