# Snake coding task

Implement a playable, single-file C11 terminal Snake game in this directory. Requirements:

- Default mode is interactive terminal play with WASD movement, `q` to quit, visible score, food, and collision/game-over handling. Use standard POSIX terminal APIs and restore terminal state on normal exit and signals.
- The game rules must be factored into deterministic functions that can be tested without a TTY. The board must have fixed dimensions, food placement must be bounded and deterministic under a seed, eating grows the snake, reverse-direction input is ignored, and wall/self collision ends the game.
- `./snake --self-test` runs deterministic assertions and exits 0 only if they all pass.
- `./snake --demo` runs a deterministic headless game and prints a stable summary containing its outcome and score; it must not require a terminal or sleep.
- `make test` compiles with strict warnings (`-std=c11 -Wall -Wextra -Wpedantic -Werror`) and runs the self-test and demo.
- Include a concise `README.md` section with build/run/test instructions. No external network or dependencies beyond a C compiler and POSIX libc.

Keep the implementation compact and safe: validate input, avoid out-of-bounds accesses, and use clear names. Use your tools to write the implementation, compile it, run tests, and fix any failures before reporting completion. Work only in this directory.