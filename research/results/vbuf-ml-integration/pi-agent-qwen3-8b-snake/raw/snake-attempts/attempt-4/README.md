# Snake coding task

Create a playable C11/POSIX terminal Snake game with no third-party dependencies. Use fixed-size arrays, a deterministic xorshift RNG, bounded food placement, WASD movement and `q` to quit. Restore terminal settings after play and signals.

Expose `--self-test` for deterministic assertions of movement, reverse-direction rejection, food growth, wall/self collisions, and seeded food placement. Expose `--demo` for a quick deterministic headless game that prints stable outcome and score. Default mode is interactive play.

Provide `make test` using `-std=c11 -Wall -Wextra -Wpedantic -Werror`; it must build, run the assertions, and run the headless demo. Document build/run/test. Use POSIX `termios`, `select`, ANSI output; no ncurses. Work only here.