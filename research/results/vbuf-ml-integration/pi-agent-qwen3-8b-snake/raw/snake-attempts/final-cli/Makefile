CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror

all: snake

snake: snake.c
	$(CC) $(CFLAGS) -o snake snake.c

test: snake
	./snake --self-test

clean:
	rm -f snake
