DIRS = cross_correlation_c cross_correlation_cuda cross_correlation_eigenshot_cuda

SUBDIRS   := $(addprefix src/, $(DIRS))
CLEANDIRS := $(addprefix clean/, $(SUBDIRS))

OBJ := $(wildcard $(addsuffix /*.o, $(addprefix build/,$(DIRS))))
LIB := $(wildcard lib/*.so)

.PHONY: subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	@+$(MAKE) -C $@

clean:
	@echo "Removing object files"
	@rm -f $(OBJ)
	@echo "Removing library files"
	@rm -f $(LIB)

realclean: clean
	@echo "Removing lib/ and build/ directories"
	@rm -rf lib/ build/
