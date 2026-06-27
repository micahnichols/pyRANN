This folder contains a number of examples of usage of mlp.
Typically, a folder contains a readme.txt file describing the example,
some input files, a sample_out folder with the expected output files,
and an empty out folder where the output files are saved.

=================================
Intergration Tests:

Any example can also serve as an automatic test of the mlip package.
To turn an example into the test, create the file test.sh in your example
folder. The purpose of the script is to check the status of the execution of
MLIP program and check out the resulting output. This script should exit with 
status of zero if the run of MLIP program were successfully parsed and 
no errors were found in the output after execution, otherwise the scipt should
exit with non-zero value.
 
