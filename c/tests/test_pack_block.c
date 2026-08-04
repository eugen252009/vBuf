#include "test_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void run_test(vbuf_instance_t *inst) {
    (void)inst;
    printf("\n--- Running Pack Block Test ---\n");

    uint32_t test_data[5] = {10, 20, 30, 40, 50};
    uint16_t key_id = 999;
    uint8_t type = 2; // maps to uint32_t, sem = 0

    // Test size-query mode (target = NULL)
    size_t needed_size = vbuf_pack_block(NULL, key_id, type, test_data, 5);
    printf("Query needed size: %zu bytes\n", needed_size);

    // Header (16) + alignment (to 4096) + data (20) + tail pad (to 8)
    // 16 to 4096 = 4096. 4096 + 20 = 4116. Aligned to 8 = 4120.
    assert(needed_size == 4120);

    void* target = malloc(needed_size);
    assert(target != NULL);

    size_t packed_size = vbuf_pack_block(target, key_id, type, test_data, 5);
    assert(packed_size == needed_size);

    // Verify packed header
    uint64_t anchor = *(uint64_t*)target;
    uint64_t count = *(uint64_t*)((uint8_t*)target + 8);
    uint16_t parsed_id = (uint16_t)((anchor >> 16) & 0xFFFF);
    uint16_t parsed_width = (uint16_t)((anchor >> 32) & 0xFFFF);
    uint8_t parsed_sem = (uint8_t)(anchor & 0xF);

    printf("Parsed Anchor: id=%u, width=%u, sem=%u, count=%lu\n", parsed_id, parsed_width, parsed_sem, count);
    assert(parsed_id == key_id);
    assert(parsed_width == 32);
    assert(parsed_sem == 0);
    assert(count == 5);

    // Verify data
    uint32_t* packed_data = (uint32_t*)((uint8_t*)target + 4096);
    for (int i = 0; i < 5; i++) {
        assert(packed_data[i] == test_data[i]);
    }

    free(target);
    printf("Pack Block Test Passed successfully!\n\n");
}
