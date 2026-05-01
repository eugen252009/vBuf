#include "vbuf.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef uint32_t target_t;
uint16_t bit_width = 32;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <num_elements>\n", argv[0]);
    return 1;
  }

  size_t n = atoll(argv[1]);
  const char *filename = "/dev/shm/dual_test.vbuf";

  // Datei zum Schreiben öffnen (Standard C)
  FILE *f = fopen(filename, "wb");
  if (!f) {
    perror("Konnte Datei nicht zum Schreiben öffnen");
    return 1;
  }

  printf("Generiere %zu Elemente für 4 Spalten...\n", n);
  // 1. DEN GLOBALEN HEADER SCHREIBEN (WICHTIG!)
  // Die 12 steht für den Shift: 1 << 12 = 4096 Alignment
  vbuf_write_header(f, 12);
  // Temporären Speicher für die Generierung reservieren
  // (Wir nutzen uint16_t, passend zu deiner vbuf_write_column)
  target_t *buffer = malloc(n * sizeof(target_t));
  if (!buffer) {
    printf("Out of memory!\n");
    fclose(f);
    return 1;
  }

  srand(time(NULL));

  // Spalte 101: Sequentiell
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)i;
  vbuf_write_atomic_column(f, 101, n, 4096, bit_width, buffer);

  // Spalte 102: Zufall
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)rand();
  vbuf_write_atomic_column(f, 102, n, 4096, bit_width, buffer);

  // Spalte 103: Index * 3
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)(i * 3);
  vbuf_write_atomic_column(f, 103, n, 4096, bit_width, buffer);

  // Spalte 104: Kleiner Zufall
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)(rand() % 100);
  vbuf_write_atomic_column(f, 104, n, 4096, bit_width, buffer);
  // Spalte 105: Index + 42
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)(i + 42);
  vbuf_write_atomic_column(f, 105, n, 4096, bit_width, buffer);

  // Spalte 106: Alternierendes Bitmuster
  for (size_t i = 0; i < n; i++)
    buffer[i] = (uint16_t)(i % 2 ? 0xAAAA : 0x5555);
  vbuf_write_atomic_column(f, 106, n, 4096, bit_width, buffer);

  free(buffer);
  fclose(f);

  printf("Datei '%s' mit 4 Spalten erfolgreich erstellt.\n", filename);
  return 0;
}
