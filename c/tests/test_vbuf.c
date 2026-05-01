#include "test_interface.h"
#include <stdio.h>

void run_test(vbuf_instance_t *inst) {
    uint8_t *curr = inst->mem + 16; // Nach Global Header
    uint8_t *end = inst->mem + inst->size;
    int cols_found = 0;

    printf("\n--- vBuf Datei-Inhaltsverzeichnis (Debug) ---\n");

    // Wir prüfen auf 16 Byte, da Anchor (8) + Count (8) jetzt das Minimum sind
    while (curr + 16 <= end) {
        uint64_t anchor = *(uint64_t *)curr;
        
        // 1. Wenn Anchor 0 ist, ist es vermutlich Alignment-Padding zwischen Blöcken
        if (anchor == 0) {
            curr += 8; 
            continue;
        }

        // 2. Felder nach deinem neuen Bit-Standard extrahieren
        uint16_t id    = (uint16_t)((anchor >> 16) & 0xFFFF);
        uint16_t plen  = (uint16_t)((anchor >> 32) & 0xFFFF);
        // Da Bit 9 (Overflow) gesetzt ist, lesen wir N aus den nächsten 8 Bytes
        uint64_t n     = *(uint64_t *)(curr + 8);

        printf("Gefunden: ID %5u | N = %10lu | %2u-Bit | Offset = %lu\n", 
                id, n, plen, (size_t)(curr - inst->mem));

        cols_found++;

        // 3. Sprung-Logik (Synchron mit vbuf.c)
        size_t header_size = 16; // Anchor(8) + Count(8)
        size_t current_pos = (size_t)(curr - inst->mem);
        
        // Sprung zum Diamond-Payload (Alignment)
        size_t data_start = (current_pos + header_size + (inst->alignment - 1)) & 
                            ~(inst->alignment - 1);
        
        // Berechnung der Datenmenge basierend auf PLen (plen/8 = bytes per item)
        size_t data_bytes = n * (plen / 8);

        // Setze curr auf das Ende der Daten
        curr = (uint8_t *)(inst->mem + data_start + data_bytes);
        
        // Nächster Anchor muss auf 8-Byte Grenze liegen (da u64)
        curr = (uint8_t *)(((uintptr_t)curr + 7) & ~7);
    }

    printf("--- Scan beendet. Insgesamt %d Spalten gefunden ---\n\n", cols_found);
}
