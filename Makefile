single_include: pretty
	mkdir -p single_include
	python3 combine.py

pretty:
	clang-format -i include/*.h include/*/*.h tests/*.cu tests/*.h examples/*/*.cu

all: single_include pretty

.PHONY: pretty single_include all

FORCE:
