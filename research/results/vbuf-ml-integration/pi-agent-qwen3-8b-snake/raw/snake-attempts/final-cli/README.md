# Snake task

Build a compact C11 turn-based terminal Snake game (no ncurses, termios, or third-party libraries). At each board display, the player enters one WASD move and presses Enter; `q` quits. Use a fixed board and fixed-size body array. Food placement must be in-bounds and never overlap the snake, with seeded deterministic RNG. Ignore immediate reverse moves; walls and body collisions end the game; eating grows and increases score.

Provide `--self-test` with deterministic assertions for movement, reverse rejection, growth, wall collision, self collision, and same-seed food placement. Provide `--demo` that runs deterministic moves without stdin/sleep and prints stable `outcome` and `score`. `make test` must compile with `-std=c11 -Wall -Wextra -Wpedantic -Werror`, run self-tests, then run demo. Include build/play/test instructions.

Use one `main`, validate all state bounds, and use only standard C/POSIX libc. Work only here.