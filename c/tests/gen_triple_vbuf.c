#include "../vbuf.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));
  const char *filename = "triple_test.vbuf";

  // 1. Datei mit Global Header initialisieren
  FILE *f = fopen(filename, "wb");
  if (!f)
    return 1;

  uint8_t a_shift = 1;
  uint32_t alignment = 16 << a_shift; // 32 Byte

  uint8_t g_hdr[16] = {0};
  *(uint32_t *)&g_hdr[0] = MAGIC;
  *(uint32_t *)&g_hdr[4] = VERSION;
  g_hdr[8] = a_shift;
  fwrite(g_hdr, 1, 16, f);

  size_t N = 2000000000ULL;

  printf("Erstelle Triple-vBuf mit Library-Funktion...\n");

  // Wir nutzen direkt die exportierten Funktionen deiner .so
  printf("-> Spalte 101 (Seq)\n");
  write_atomic_column(f, 101, N, alignment, 0); // 0 = Seq Mode in deiner Lib?

  printf("-> Spalte 102 (Random)\n");
  write_atomic_column(f, 102, N, alignment, 1); // 1 = Random Mode

  printf("-> Spalte 103 (Payload)\n");
  write_atomic_column(f, 103, N, alignment, 2); // 2 = Const Mode

  fclose(f);
  printf("Fertig!\n");
  return 0;
}
