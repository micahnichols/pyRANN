#/bin/bash

# Preamble, common for all examples
MLP_EXE=../../../bin/mlp
TMP_DIR=./out
mkdir -p $TMP_DIR

# Body:
$MLP_EXE relax pot.almtp to-relax.cfg $TMP_DIR/relaxed.cfg --relaxation_settings=relax.ini --extrapolation_control --save_extrapolative_to=$TMP_DIR/preselected.cfg --threshold_save=2 --threshold_break=10 
