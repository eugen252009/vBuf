CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror

all: snake

test: snake
	./snake --self-test
	./snake --demo

demo: snake
	./snake --demo

snake: snake.c
	$(CC) $(CFLAGS) -o snake snake.c
