import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin')))

import mlippy as mlp

mlp.initialize("mlippy-test");

if not os.path.exists('./out'):
	os.makedirs('./out')
else: 
	if os.path.isfile("./out/trained.cfg"):
		os.remove('./out/trained.mtp')

mlp.mlp_selftest()
mlp.finalize()
