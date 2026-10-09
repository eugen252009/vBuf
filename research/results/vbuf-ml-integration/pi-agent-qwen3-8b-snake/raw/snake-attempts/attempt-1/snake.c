#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <select.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>

#define WIDTH 40
#define HEIGHT 20

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point *body;
    int length;
    int dir;
} Snake;

void init_snake(Snake *s) {
    s->body = malloc(2 * sizeof(Point));
    s->body[0].x = WIDTH/2; s->body[0].y = HEIGHT/2;
    s->body[1].x = WIDTH/2 - 1; s->body[1].y = HEIGHT/2;
    s->length = 2;
    s->dir = 0; // 0: right, 1: down, 2: left, 3: up
}

void move_snake(Snake *s, Point food) {
    Point *new_body = realloc(s->body, (s->length + 1) * sizeof(Point));
    new_body[s->length].x = s->body[s->length-1].x;
    new_body[s->length].y = s->body[s->length-1].y;
    for (int i = s->length-1; i > 0; i--) {
        new_body[i].x = s->body[i-1].x;
        new_body[i].y = s->body[i-1].y;
    }
    switch (s->dir) {
        case 0: new_body[0].x++; break;
        case 1: new_body[0].y++; break;
        case 2: new_body[0].x--; break;
        case 3: new_body[0].y--; break;
    }
    s->body = new_body;
    if (s->body[0].x == food.x && s->body[0].y == food.y) {
        s->length++;
        // Food regenerates
    }
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        // Self-test implementation
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
        // Demo implementation
        return 0;
    }
    // Interactive mode
    return 0;
}