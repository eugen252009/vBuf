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
CONFORMANCE_EXE = v05_compat

# Scannt alle test_*.c Dateien
TEST_SOURCES = $(wildcard tests/test_*.c)
ACTIVE_TESTS = $(patsubst tests/%.c, active_tests/%.so, $(TEST_SOURCES))

all: $(LIB_NAME) $(RUNNER_EXE) vbuf_writer tests_dir $(ACTIVE_TESTS)

# 1. Die Haupt-Library (von Rust gebaut und herüberkopiert)
$(LIB_NAME): ../rust/src/lib.rs ../rust/Cargo.toml
	cd ../rust && cargo build --release
	cp ../rust/target/release/libvbuf_core.so ./$(LIB_NAME)

# 2. Der Runner
$(RUNNER_EXE): $(RUNNER_SRC) vbuf.h $(LIB_NAME)
	$(CC) $(CFLAGS) $(RUNNER_SRC) $(LDFLAGS) -ldl -lvbuf -o $(RUNNER_EXE)

# 3. Der Writer (verwendet das verlinkte libvbuf.so)
vbuf_writer: vBuf_writer.c vbuf.h $(LIB_NAME)
	$(CC) $(CFLAGS) vBuf_writer.c $(LDFLAGS) -lvbuf -o vbuf_writer

# 4. Universal-Regel für Plugins
active_tests/%.so: tests/%.c tests/test_interface.h $(LIB_NAME)
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -lvbuf $< -o $@

# 5. Cross-language legacy fixture evidence (not a v0.6 contract)
$(CONFORMANCE_EXE): conformance/v05_compat.c vbuf.h $(LIB_NAME)
	$(CC) $(CFLAGS) conformance/v05_compat.c $(LDFLAGS) -lvbuf -o $(CONFORMANCE_EXE)

conformance: $(CONFORMANCE_EXE)
	./$(CONFORMANCE_EXE) ../tests/fixtures/v05

tests_dir:
	@mkdir -p ./active_tests

test: all
	@if [ ! -f dual_test.vbuf ]; then ./vbuf_writer 100000000; fi
	@for plugin in $(ACTIVE_TESTS); do \
		echo "\n\033[1;34m>>> Executing $$plugin ...\033[0m"; \
		./$(RUNNER_EXE) /dev/shm/dual_test.vbuf $$plugin; \
	done

clean:
	rm -rf *.so $(RUNNER_EXE) $(CONFORMANCE_EXE) vbuf_writer ./active_tests
