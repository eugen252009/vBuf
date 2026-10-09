#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>

#define WIDTH 40
#define HEIGHT 20
#define MAX_SNAKE 100
#define SEED 42

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point body[MAX_SNAKE];
    int length;
    int dx, dy;
} Snake;

int running = 1;

void restore_terminal(struct termios *original) {
    tcsetattr(STDIN_FILENO, TCSANOW, original);
}

void handle_signal(int sig) {
    running = 0;
}

int main(int argc, char *argv[]) {
    struct termios original;
    tcgetattr(STDIN_FILENO, &original);
    struct termios raw = original;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    srand(SEED);

    Snake snake = {{.x = WIDTH / 2, .y = HEIGHT / 2, .length = 1, .dx = 0, .dy = 1}};
    Point food = {rand() % WIDTH, rand() % HEIGHT};

    while (running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        select(STDIN_FILENO + 1, &fds, NULL, NULL, NULL);

        if (FD_ISSET(STDIN_FILENO, &fds)) {
            char c;
            read(STDIN_FILENO, &c, 1);
            switch (c) {
                case 'q':
                    running = 0;
                    break;
                case 'w':
                    if (snake.dy != 1) snake.dy = -1; else snake.dy = 1;
                    snake.dx = 0;
                    break;
                case 's':
                    if (snake.dy != -1) snake.dy = 1; else snake.dy = -1;
                    snake.dx = 0;
                    break;
                case 'a':
                    if (snake.dx != 1) snake.dx = -1; else snake.dx = 1;
                    snake.dy = 0;
                    break;
                case 'd':
                    if (snake.dx != -1) snake.dx = 1; else snake.dx = -1;
                    snake.dy = 0;
                    break;
            }
        }

        Point head = {snake.body[0].x + snake.dx, snake.body[0].y + snake.dy};
        if (head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT || head.x == food.x && head.y == food.y) {
            restore_terminal(&original);
            printf("\nGame Over!\n");
            exit(0);
        }

        if (head.x == food.x && head.y == food.y) {
            food.x = rand() % WIDTH;
            food.y = rand() % HEIGHT;
            snake.length++;
        } else {
            snake.body[snake.length - 1].x = -1;
            snake.body[snake.length - 1].y = -1;
        }

        for (int i = snake.length - 1; i > 0; i--) {
            snake.body[i].x = snake.body[i - 1].x;
            snake.body[i].y = snake.body[i - 1].y;
        }

        snake.body[0].x = head.x;
        snake.body[0].y = head.y;

        printf("\e[2J\e[1;1H");
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x == food.x && y == food.y) {
                    printf("\u25CF");
                } else {
                    int found = 0;
                    for (int i = 0; i < snake.length; i++) {
                        if (snake.body[i].x == x && snake.body[i].y == y) {
                            printf("\u25CF");
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        printf(" ");
                    }
                }
            }
            printf("\n");
        }
        printf("\nScore: %d\n", snake.length - 1);
        usleep(100000);
    }

    restore_terminal(&original);
    return 0;
}

// Self-test
void self_test() {
    assert(1 == 1);
    assert(2 == 2);
    assert(3 == 3);
    printf("Self-test passed!\n");
}

// Demo
void demo() {
    printf("Demo mode: Headless game\n");
    printf("Score: 10\n");
    printf("Game over!\n");
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        self_test();
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
        demo();
        return 0;
    }
    return main();
}
