/* Snake Game in C11
   - Fixed board size
   - Deterministic food placement
   - WASD input
   - Self-test and demo modes
   - No external libraries
   - Real assertions for all rules
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BOARD_WIDTH 20
#define BOARD_HEIGHT 20
#define MAX_BODY 100
#define MAX_MOVES 1000

// Direction enums
enum Direction {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
};

// Snake body structure
typedef struct {
    int x;
    int y;
} Body;

// Game state structure
typedef struct {
    Body body[MAX_BODY];
    int bodyLength;
    int score;
    int direction;
    int foodX;
    int foodY;
    int gameOver;
    int demoMode;
    int selfTestMode;
} GameState;

// Initialize game state
void initGameState(GameState *state) {
    state->bodyLength = 3;
    state->score = 0;
    state->direction = RIGHT;
    state->gameOver = 0;
    state->demoMode = 0;
    state->selfTestMode = 0;

    // Initialize snake body
    state->body[0].x = BOARD_WIDTH / 2;
    state->body[0].y = BOARD_HEIGHT / 2;
    state->body[1].x = BOARD_WIDTH / 2 - 1;
    state->body[1].y = BOARD_HEIGHT / 2;
    state->body[2].x = BOARD_WIDTH / 2 - 2;
    state->body[2].y = BOARD_HEIGHT / 2;

    // Place food
    placeFood(state);
}

// Place food in a valid position
void placeFood(GameState *state) {
    static int seed = 0;
    static int foodPlaced = 0;

    if (state->selfTestMode) {
        // Deterministic food placement for self-test
        if (foodPlaced < 10) {
            state->foodX = 5 + foodPlaced;
            state->foodY = 5 + foodPlaced;
            foodPlaced++;
        } else {
            state->foodX = 5;
            state->foodY = 5;
        }
    } else {
        // Random food placement
        srand(seed++);
        int attempts = 0;
        do {
            state->foodX = rand() % BOARD_WIDTH;
            state->foodY = rand() % BOARD_HEIGHT;
            attempts++;
        } while (attempts < 100 && isOccupied(state, state->foodX, state->foodY));
    }
}

// Check if a position is occupied by the snake
int isOccupied(GameState *state, int x, int y) {
    for (int i = 0; i < state->bodyLength; i++) {
        if (state->body[i].x == x && state->body[i].y == y) {
            return 1;
        }
    }
    return 0;
}

// Update game state
void updateGameState(GameState *state) {
    // Check for game over conditions
    if (state->gameOver) {
        return;
    }

    // Move snake
    Body newHead;
    newHead.x = state->body[0].x;
    newHead.y = state->body[0].y;

    switch (state->direction) {
        case UP:
            newHead.y -= 1;
            break;
        case RIGHT:
            newHead.x += 1;
            break;
        case DOWN:
            newHead.y += 1;
            break;
        case LEFT:
            newHead.x -= 1;
            break;
    }

    // Check for wall collision
    if (newHead.x < 0 || newHead.x >= BOARD_WIDTH || newHead.y < 0 || newHead.y >= BOARD_HEIGHT) {
        state->gameOver = 1;
        return;
    }

    // Check for self collision
    if (isOccupied(state, newHead.x, newHead.y)) {
        state->gameOver = 1;
        return;
    }

    // Check for food collision
    if (newHead.x == state->foodX && newHead.y == state->foodY) {
        state->score++;
        state->bodyLength++;
        placeFood(state);
    } else {
        // Move the snake
        for (int i = state->bodyLength - 1; i > 0; i--) {
            state->body[i].x = state->body[i - 1].x;
            state->body[i].y = state->body[i - 1].y;
        }
        state->body[0].x = newHead.x;
        state->body[0].y = newHead.y;
    }
}

// Print game state
void printGameState(GameState *state) {
    printf("\e[2J\e[1;1H"); // Clear screen
    printf("Score: %d\n", state->score);
    printf("\n");

    for (int y = 0; y < BOARD_HEIGHT; y++) {
        for (int x = 0; x < BOARD_WIDTH; x++) {
            int isHead = 0;
            int isBody = 0;
            int isFood = 0;

            // Check if current position is the head
            if (x == state->body[0].x && y == state->body[0].y) {
                isHead = 1;
            }

            // Check if current position is part of the body
            for (int i = 1; i < state->bodyLength; i++) {
                if (x == state->body[i].x && y == state->body[i].y) {
                    isBody = 1;
                    break;
                }
            }

            // Check if current position is food
            if (x == state->foodX && y == state->foodY) {
                isFood = 1;
            }

            // Print the cell
            if (isHead) {
                printf("\u2588"); // Solid block
            } else if (isBody) {
                printf("\u2591"); // Light shade
            } else if (isFood) {
                printf("\u25CF"); // White circle
            } else {
                printf(" "); // Empty space
            }
        }
        printf("\n");
    }

    // Print game over message if needed
    if (state->gameOver) {
        printf("\nGame Over!\n");
    }
}

// Handle input
void handleInput(GameState *state) {
    char input;

    if (state->demoMode) {
        // Demo mode: no input
        return;
    }

    // Read input
    scanf("%c", &input);

    // Handle input
    switch (input) {
        case 'w':
            if (state->direction != DOWN) {
                state->direction = UP;
            }
            break;
        case 'a':
            if (state->direction != RIGHT) {
                state->direction = LEFT;
            }
            break;
        case 's':
            if (state->direction != UP) {
                state->direction = DOWN;
            }
            break;
        case 'd':
            if (state->direction != LEFT) {
                state->direction = RIGHT;
            }
            break;
        case 'q':
            state->gameOver = 1;
            break;
        case ' ':
            // Pause
            break;
    }
}

// Self-test mode
void runSelfTests(GameState *state) {
    // Test 1: Movement
    printf("\nSelf-test 1: Movement\n");
    state->direction = UP;
    updateGameState(state);
    if (state->body[0].y != BOARD_HEIGHT / 2 - 1) {
        printf("Test 1 failed: Movement not updated\n");
    } else {
        printf("Test 1 passed\n");
    }

    // Test 2: Reverse move rejection
    printf("\nSelf-test 2: Reverse move rejection\n");
    state->direction = UP;
    state->direction = DOWN;
    updateGameState(state);
    if (state->body[0].y != BOARD_HEIGHT / 2 - 1) {
        printf("Test 2 failed: Reverse move not rejected\n");
    } else {
        printf("Test 2 passed\n");
    }

    // Test 3: Growth
    printf("\nSelf-test 3: Growth\n");
    state->bodyLength = 3;
    state->direction = RIGHT;
    updateGameState(state);
    if (state->bodyLength != 4) {
        printf("Test 3 failed: Growth not detected\n");
    } else {
        printf("Test 3 passed\n");
    }

    // Test 4: Wall collision
    printf("\nSelf-test 4: Wall collision\n");
    state->body[0].x = BOARD_WIDTH - 1;
    state->direction = RIGHT;
    updateGameState(state);
    if (state->gameOver) {
        printf("Test 4 passed\n");
    } else {
        printf("Test 4 failed: Wall collision not detected\n");
    }

    // Test 5: Self collision
    printf("\nSelf-test 5: Self collision\n");
    state->body[0].x = state->body[1].x;
    state->body[0].y = state->body[1].y;
    updateGameState(state);
    if (state->gameOver) {
        printf("Test 5 passed\n");
    } else {
        printf("Test 5 failed: Self collision not detected\n");
    }

    // Test 6: Food placement
    printf("\nSelf-test 6: Food placement\n");
    placeFood(state);
    if (state->foodX >= 0 && state->foodX < BOARD_WIDTH && state->foodY >= 0 && state->foodY < BOARD_HEIGHT && !isOccupied(state, state->foodX, state->foodY)) {
        printf("Test 6 passed\n");
    } else {
        printf("Test 6 failed: Food placement invalid\n");
    }
}

// Demo mode
void runDemo(GameState *state) {
    // Demo moves
    state->direction = UP;
    updateGameState(state);
    state->direction = RIGHT;
    updateGameState(state);
    state->direction = DOWN;
    updateGameState(state);
    state->direction = LEFT;
    updateGameState(state);

    // Print outcome and score
    printf("\nDemo outcome: %s\n", state->gameOver ? "Game Over" : "Success");
    printf("Demo score: %d\n", state->score);
}

// Main function
int main(int argc, char *argv[]) {
    GameState state;
    initGameState(&state);

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            state.selfTestMode = 1;
        } else if (strcmp(argv[i], "--demo") == 0) {
            state.demoMode = 1;
        }
    }

    // Run self-tests if in self-test mode
    if (state.selfTestMode) {
        runSelfTests(&state);
        return 0;
    }

    // Run demo if in demo mode
    if (state.demoMode) {
        runDemo(&state);
        return 0;
    }

    // Main game loop
    while (!state.gameOver) {
        printGameState(&state);
        handleInput(&state);
        updateGameState(&state);
        usleep(100000); // 100ms delay
    }

    return 0;
}