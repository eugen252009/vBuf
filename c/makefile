CC = gcc
CFLAGS = -Ofast -ffast-math -flto -Wall -Wextra -fPIC -fstrict-aliasing -fopenmp
LIB_FLAGS = -shared -fPIC
LDFLAGS = -L. -Wl,-rpath,. -lm -fopenmp

# Architektur-Check
ARCH := $(shell uname -m)
ifeq ($(ARCH), x86_64)
    CFLAGS += -march=native
endif

LIB_NAME = libvbuf.so
RUNNER_SRC = tests/vbuf_test_runner.c
RUNNER_EXE = vbuf_test_runner

# Scannt alle test_*.c Dateien
TEST_SOURCES = $(wildcard tests/test_*.c)
ACTIVE_TESTS = $(patsubst tests/%.c, active_tests/%.so, $(TEST_SOURCES))

all: $(LIB_NAME) $(RUNNER_EXE) vbuf_writer tests_dir $(ACTIVE_TESTS)

# 1. Die Haupt-Library
$(LIB_NAME): vbuf.c vbuf.h
	$(CC) $(CFLAGS) -shared vbuf.c -o $(LIB_NAME)

# 2. Der Runner
$(RUNNER_EXE): $(RUNNER_SRC) vbuf.h $(LIB_NAME)
	$(CC) $(CFLAGS) $(RUNNER_SRC) $(LDFLAGS) -ldl -lvbuf -o $(RUNNER_EXE)

# 3. Der Writer (GEÄNDERT: vbuf.c wird direkt mit einkompiliert)
vbuf_writer: vBuf_writer.c vbuf.c vbuf.h
	$(CC) $(CFLAGS) vBuf_writer.c vbuf.c -lm -fopenmp -o vbuf_writer

# 4. Universal-Regel für Plugins
active_tests/%.so: tests/%.c tests/test_interface.h $(LIB_NAME)
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -lvbuf $< -o $@

tests_dir:
	@mkdir -p ./active_tests

test: all
	@if [ ! -f dual_test.vbuf ]; then ./vbuf_writer 100000000; fi
	@for plugin in $(ACTIVE_TESTS); do \
		echo "\n\033[1;34m>>> Executing $$plugin ...\033[0m"; \
		./$(RUNNER_EXE) /dev/shm/dual_test.vbuf $$plugin; \
	done

clean:
	rm -rf *.so $(RUNNER_EXE) vbuf_writer ./active_tests
