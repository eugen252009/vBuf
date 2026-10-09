CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror -g

all: snake

snake: snake.c
	$(CC) $(CFLAGS) -o snake snake.c

self-test: snake
	./snake --self-test

demo: snake
	./snake --demo

test: self-test demo

clean:
	rm -f snake

# Play instructions
# To play:
# 1. Run ./snake
# 2. Use WASD to move
# 3. Press 'q' to quit
# 4. Press ' ' to pause

# To run self-tests:
# make self-test

# To run demo:
# make demo