#
# TESTING
# Integration testing
#   

# configuration makefile
-include ../make/config.mk
#CURDIR=.

RUN_EXAMPLE=test.sh

BIN_DIR = $(abspath $(CURDIR)/../bin)
SELFTEST_DIR = $(wildcard $(CURDIR)/self-test)
EXAMPLE_SCRIPTS = $(sort $(wildcard $(CURDIR)/examples/**/$(RUN_EXAMPLE)) $(wildcard $(CURDIR)/examples/**/$(RUN_EXAMPLE)))

ifeq ($(PYTHON_EXISTS), 1)
.PHONY: all
all: UNIT_TEST TEST_HEADER $(EXAMPLE_SCRIPTS) PYTHON_TEST 
else
.PHONY: all
all: UNIT_TEST TEST_HEADER $(EXAMPLE_SCRIPTS) 
endif

.PHONY: clean
clean: $(sort $(wildcard $(CLEAN_DIR)/**/out))
	@ $(RM) -r $^

ifneq ($(USE_MPI), 1)
.PHONY: TEST_HEADER
TEST_HEADER:
	@echo SERIAL INTEGRATION TESTING:
else
.PHONY: TEST_HEADER
TEST_HEADER:
	@echo PARALLEL INTEGRATION TESTING:
endif

.PHONY: $(EXAMPLE_SCRIPTS) 
$(EXAMPLE_SCRIPTS):
	@echo Running test $(notdir $(realpath $(dir $@)))
	@cd $(dir $@); \
	 chmod +x *.sh; \
	./$(RUN_EXAMPLE)>/dev/null
		
#$@ 1>out/stdout.log

ifneq ($(USE_MPI), 1)
.PHONY: UNIT_TEST
UNIT_TEST:
	@echo Running serial self test
	@mkdir -p $(SELFTEST_DIR)/temp
	@cd $(SELFTEST_DIR); $(BIN_DIR)/mlp test
else
.PHONY: UNIT_TEST
UNIT_TEST:
	@echo Running parallel self test
	@mkdir -p $(SELFTEST_DIR)/temp
	@cd $(SELFTEST_DIR); mpirun -n 3 $(BIN_DIR)/mlp test #rm temp/*; sleep 1; 
endif

