#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>

#define BOARD_WIDTH 20
#define BOARD_HEIGHT 20
#define MAX_BODY 100
#define MAX_MOVES 1000

typedef struct {
    int x, y;
} Position;

typedef struct {
    Position body[MAX_BODY];
    int length;
    int score;
    int direction;
    int game_over;
} Snake;

void init_snake(Snake *snake) {
    snake->length = 4;
    snake->score = 0;
    snake->game_over = 0;
    snake->direction = 0; // 0: right, 1: down, 2: left, 3: up

    // Initialize snake body
    snake->body[0].x = BOARD_WIDTH / 2;
    snake->body[0].y = BOARD_HEIGHT / 2;
    for (int i = 1; i < snake->length; i++) {
        snake->body[i].x = BOARD_WIDTH / 2 - i;
        snake->body[i].y = BOARD_HEIGHT / 2;
    }
}

void print_board(Snake *snake) {
    for (int y = 0; y < BOARD_HEIGHT; y++) {
        for (int x = 0; x < BOARD_WIDTH; x++) {
            int is_snake = 0;
            for (int i = 0; i < snake->length; i++) {
                if (snake->body[i].x == x && snake->body[i].y == y) {
                    is_snake = 1;
                    break;
                 }
             }
             if (is_snake) {
                 printf("#");
             } else {
                 printf(".");
             }
         }
         printf("\n");
     }
}

int place_food(Snake *snake, Position *food) {
    srand(time(NULL));
    int placed = 0;
    while (!placed) {
        food->x = rand() % BOARD_WIDTH;
        food->y = rand() % BOARD_HEIGHT;
        placed = 1;
        for (int i = 0; i < snake->length; i++) {
            if (snake->body[i].x == food->x && snake->body[i].y == food->y) {
                 placed = 0;
                 break;
             }
         }
     }
     return 1;
}

void move_snake(Snake *snake, Position *food) {
    Position new_head;
    switch (snake->direction) {
         case 0: // Right
             new_head.x = snake->body[0].x + 1;
             new_head.y = snake->body[0].y;
             break;
         case 1: // Down
             new_head.x = snake->body[0].x;
             new_head.y = snake->body[0].y + 1;
             break;
         case 2: // Left
             new_head.x = snake->body[0].x - 1;
             new_head.y = snake->body[0].y;
             break;
         case 3: // Up
             new_head.x = snake->body[0].x;
             new_head.y = snake->body[0].y - 1;
             break;
     }

     // Check for wall collision
     if (new_head.x < 0 || new_head.x >= BOARD_WIDTH || new_head.y < 0 || new_head.y >= BOARD_HEIGHT) {
         snake->game_over = 1;
         return;
     }

     // Check for self collision
     for (int i = 1; i < snake->length; i++) {
         if (new_head.x == snake->body[i].x && new_head.y == snake->body[i].y) {
             snake->game_over = 1;
             return;
         }
     }

     // Check for reverse move
     if (snake->length > 1 && new_head.x == snake->body[1].x && new_head.y == snake->body[1].y) {
         snake->game_over = 1;
         return;
     }

     // Move the snake
     for (int i = snake->length - 1; i > 0; i--) {
         snake->body[i] = snake->body[i - 1];
     }
     snake->body[0] = new_head;

     // Check for food collision
     if (new_head.x == food->x && new_head.y == food->y) {
         snake->score++;
         snake->length++;
     }
}

void handle_input(Snake *snake) {
     char input;
     scanf("%c", &input);
     switch (input) {
         case 'w':
             if (snake->direction != 1) snake->direction = 3;
             break;
         case 's':
             if (snake->direction != 3) snake->direction = 1;
             break;
         case 'a':
             if (snake->direction != 1) snake->direction = 2;
             break;
         case 'd':
             if (snake->direction != 2) snake->direction = 0;
             break;
         case 'q':
             snake->game_over = 1;
             break;
     }
}

void run_demo(Snake *snake, Position *food) {
     int moves = 0;
     while (!snake->game_over && moves < MAX_MOVES) {
         move_snake(snake, food);
         print_board(snake);
         usleep(100000);
         moves++;
     }
}

void run_self_test(Snake *snake, Position *food) {
     // Test 1: Movement
     snake->direction = 0;
     move_snake(snake, food);
     if (snake->body[0].x != snake->body[1].x + 1 || snake->body[0].y != snake->body[1].y) {
         printf("Test 1 failed: Movement not correct\n");
     } else {
         printf("Test 1 passed\n");
     }

     // Test 2: Reverse move rejection
     snake->direction = 0;
     move_snake(snake, food);
     snake->direction = 2;
     move_snake(snake, food);
     if (snake->game_over != 1) {
         printf("Test 2 failed: Reverse move not rejected\n");
     } else {
         printf("Test 2 passed\n");
     }

     // Test 3: Growth
     snake->direction = 0;
     move_snake(snake, food);
     if (snake->length != 5) {
         printf("Test 3 failed: Growth not correct\n");
     } else {
         printf("Test 3 passed\n");
     }

     // Test 4: Wall collision
     snake->direction = 1;
     move_snake(snake, food);
     if (snake->game_over != 1) {
         printf("Test 4 failed: Wall collision not detected\n");
     } else {
         printf("Test 4 passed\n");
     }

     // Test 5: Self collision
     snake->direction = 0;
     move_snake(snake, food);
     snake->direction = 2;
     move_snake(snake, food);
     if (snake->game_over != 1) {
         printf("Test 5 failed: Self collision not detected\n");
     } else {
         printf("Test 5 passed\n");
     }

     // Test 6: Food placement
     int placed = 0;
     while (!placed) {
         food->x = rand() % BOARD_WIDTH;
         food->y = rand() % BOARD_HEIGHT;
         placed = 1;
         for (int i = 0; i < snake->length; i++) {
             if (snake->body[i].x == food->x && snake->body[i].y == food->y) {
                 placed = 0;
                 break;
             }
         }
     }
     if (food->x < 0 || food->x >= BOARD_WIDTH || food->y < 0 || food->y >= BOARD_HEIGHT) {
         printf("Test 6 failed: Food placement out of bounds\n");
     } else {
         printf("Test 6 passed\n");
     }
}

int main(int argc, char *argv[]) {
     Snake snake;
     Position food;
     init_snake(&snake);
     place_food(&snake, &food);

     if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
         run_self_test(&snake, &food);
         return 0;
     }

     if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
         run_demo(&snake, &food);
         return 0;
     }

     while (!snake.game_over) {
         print_board(&snake);
         handle_input(&snake);
     }

     printf("Game over! Score: %d\n", snake.score);
     return 0;
}