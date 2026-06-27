#
# mlp: generates
#   $(BIN_DIR)/mlp
#
# MORE
#   $(LIB_DIR)/lib_mlip_interface.a
#   $(LIB_DIR)/lib_mlip_cblas.a

.SECONDEXPANSION:
.SUFFIXES:

CPPFLAGS += -MMD

# configuration makefile
-include make/config.mk

# source-file directories
SRC_DIR = src
SRC_DEV_DIR = dev_src

# source files, all that can be included in all targets
SRC := $(wildcard $(SRC_DIR)/common/*.cpp) 
SRC += $(wildcard $(SRC_DIR)/drivers/*.cpp)
SRC += $(wildcard $(SRC_DIR)/test_suite/*.cpp)
SRC += $(wildcard $(SRC_DIR)/*.cpp) 
SRC += $(wildcard $(SRC_DIR)/mlp/*.cpp) 

#SRC += $(wildcard $(SRC_DEV_DIR)/*.cpp $(SRC_DEV_DIR)/mlp/*.cpp) 

#SRC := $(filter-out $(SRC_DIR)/mlp/mlp.cpp $(SRC_DEV_DIR)/mlp/mlp.cpp, $(SRC))
SRC := $(filter-out $(SRC_DIR)/mlp/mlp.cpp, $(SRC))

OBJ_MPI += $(SRC)
OBJ_MPI := $(OBJ_MPI:$(SRC_DIR)/%=$(OBJ_DIR)/mpi/$(SRC_DIR)/%.o)
#OBJ_MPI := $(OBJ_MPI:$(SRC_DEV_DIR)/%=$(OBJ_DIR)/mpi/$(SRC_DEV_DIR)/%.o)

OBJ_SERIAL += $(SRC)
OBJ_SERIAL := $(OBJ_SERIAL:$(SRC_DIR)/%=$(OBJ_DIR)/serial/$(SRC_DIR)/%.o)
#OBJ_SERIAL := $(OBJ_SERIAL:$(SRC_DEV_DIR)/%=$(OBJ_DIR)/serial/$(SRC_DEV_DIR)/%.o)

SRC_BLAS := $(wildcard blas/*.f)
SRC_BLAS += $(wildcard blas/cblas/*.c)
SRC_BLAS += $(wildcard blas/cblas/*.f)

OBJ_BLAS_SERIAL += $(SRC_BLAS)
OBJ_BLAS_SERIAL := $(OBJ_BLAS_SERIAL:blas/%=$(OBJ_DIR)/serial/blas/%.o)
OBJ_BLAS_MPI += $(SRC_BLAS)
OBJ_BLAS_MPI := $(OBJ_BLAS_MPI:blas/%=$(OBJ_DIR)/mpi/blas/%.o)

# Rule for creating directories; .PRECIOUS for not attempting stupid deletions upon completing make
.PRECIOUS: %/
%/:
	@mkdir -p $@

$(OBJ_DIR)/mpi/%.cpp.o: %.cpp | $$(dir $(OBJ_DIR)/mpi/%)/
	$(CXX_MPI) $(CPPFLAGS) $(CPPFLAGS_MPI) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/mpi/%.c.o: %.c | $$(dir $(OBJ_DIR)/mpi/%)/
	$(CC_MPI) $(CPPFLAGS_MPI) $(CCFLAGS) -c $< -o $@
$(OBJ_DIR)/mpi/%.f90.o: %.f90 | $$(dir $(OBJ_DIR)/mpi/%)/
	$(FC_MPI) $(FFLAGS) -c $< -o $@
$(OBJ_DIR)/mpi/%.f.o: %.f | $$(dir $(OBJ_DIR)/mpi/%)/
	$(FC_MPI) $(FFLAGS) -c $< -o $@
$(OBJ_DIR)/serial/%.cpp.o: %.cpp | $$(dir $(OBJ_DIR)/serial/%)/
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/serial/%.c.o: %.c | $$(dir $(OBJ_DIR)/serial/%)/
	$(CC) $(CPPFLAGS) $(CCFLAGS) -c $< -o $@
$(OBJ_DIR)/serial/%.f90.o: %.f90 | $$(dir $(OBJ_DIR)/serial/%)/
	$(FC) $(FFLAGS) -c $< -o $@
$(OBJ_DIR)/serial/%.f.o: %.f | $$(dir $(OBJ_DIR)/serial/%)/
	$(FC) $(FFLAGS) -c $< -o $@

# -----------------   BEGIN mlp   -----------------
.PHONY: mlp 
mlp: $(BIN_DIR)/mlp

#ifneq ($(USE_MLIP_PUBLIC), 1)
#    MLP_DIR = $(SRC_DEV_DIR)
#else
    MLP_DIR = $(SRC_DIR)
#endif

ifeq ($(USE_MPI), 1)

$(BIN_DIR)/mlp: $(OBJ_DIR)/mpi/$(MLP_DIR)/mlp/mlp.cpp.o $(OBJ_MPI) $(PREREQ) | $(BIN_DIR)/
	$(CXX_MPI) $^ $(LDFLAGS) $(LDLIBS) -o $@

else

$(BIN_DIR)/mlp: $(OBJ_DIR)/serial/$(MLP_DIR)/mlp/mlp.cpp.o $(OBJ_SERIAL) $(PREREQ) | $(BIN_DIR)/
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

endif
# -----------------   END mlp   -----------------
	
ifneq (1, $(USE_MPI))
# -----------------   BEGIN libinterface   -----------------
.PHONY: libinterface
libinterface: $(LIB_DIR)/lib_mlip_interface.a

OBJ_INTERFACE += $(OBJ_DIR)/serial/$(SRC_DIR)/external/interface.cpp.o
ifneq (1, $(words $(filter -DMLIP_INTEL_MKL, $(CXXFLAGS))))
  OBJ_INTERFACE += $(OBJ_BLAS_SERIAL)
endif

$(LIB_DIR)/lib_mlip_interface.a: $(OBJ_SERIAL) $(OBJ_INTERFACE) $(PREREQ) | $(LIB_DIR)/
	$(AR) -rs $@ $^
else
.PHONY: libinterface
libinterface: $(LIB_DIR)/lib_mlip_interface.a

OBJ_INTERFACE += $(OBJ_DIR)/mpi/$(SRC_DIR)/external/interface.cpp.o
ifneq (1, $(words $(filter -DMLIP_INTEL_MKL, $(CXXFLAGS))))
  OBJ_INTERFACE += $(OBJ_BLAS_MPI)
endif

$(LIB_DIR)/lib_mlip_interface.a: $(OBJ_MPI) $(OBJ_INTERFACE) $(PREREQ) | $(LIB_DIR)/
	$(AR) -rs $@ $^
endif
# -----------------   END libinterface   -----------------

# -----------------   BEGIN cblas   -----------------
cblas: $(LIB_DIR)/lib_mlip_cblas.a
SRC_BLAS := $(wildcard blas/*.f)
SRC_BLAS += $(wildcard blas/cblas/*.c)
SRC_BLAS += $(wildcard blas/cblas/*.f)
OBJ_BLAS += $(SRC_BLAS)
OBJ_BLAS := $(OBJ_BLAS:blas/%=$(OBJ_DIR)/serial/blas/%.o)

$(LIB_DIR)/lib_mlip_cblas.a: $(OBJ_BLAS) | $(LIB_DIR)/
	$(AR) -rs $@ $^

# -----------------   END cblas   -----------------

# ----------------   BEGIN libmlip ----------------
.PHONY: libmlip
libmlip: $(LIB_DIR)/lib_mlip.a

ifeq ($(USE_MPI), 1)
$(LIB_DIR)/lib_mlip.a: $(OBJ_MPI) | $(LIB_DIR)/
	$(AR) -rs $@ $^
else
$(LIB_DIR)/lib_mlip.a: $(OBJ_SERIAL) | $(LIB_DIR)/
	$(AR) -rs $@ $^
endif
# -----------------   END libmlip   -----------------

NODEPS += clean distclean help

clean: 
	@for f in mlp; do $(RM) "./bin/$$f" ; done
	@find $(BIN_DIR) -type d -empty -delete 2>/dev/null || true
	@$(RM) -rf $(OBJ_DIR)
	@$(RM) -rf $(LIB_DIR)

.PHONY: clean-cfg
clean-cfg:
	@ $(RM) -f make/config.mk

.PHONY: test
test:
	@ $(MAKE) --no-print-directory -C ./test

.PHONY: clean-test
clean-test:
	@ $(MAKE) --no-print-directory -C ./test clean

distclean: clean-cfg clean-test

# including dependencies in one go
ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
-include $(shell find ./$(OBJ_DIR) -name "*.d")
endif

# this is needed for ./configure to check what features does make have
.PHONY: has-feature
has-feature:
	@echo $(findstring $(WHAT), $(.FEATURES))
