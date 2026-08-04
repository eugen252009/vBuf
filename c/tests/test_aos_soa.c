#include "test_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct SensorData {
    uint32_t device_id;
    float temperature;
    uint16_t status_code;
};

void run_test(vbuf_instance_t *inst) {
    (void)inst;
    printf("\n--- Running AoS-to-SoA Integration Test ---\n");

    // 1. Create dummy row-oriented (AoS) data
    #define NUM_ROWS 5
    struct SensorData rows[NUM_ROWS] = {
        {1001, 23.5f, 0},
        {1002, 24.1f, 0},
        {1003, 19.8f, 1},
        {1004, 20.0f, 0},
        {1005, 35.2f, 2}
    };

    // Extract fields into contiguous arrays for packing
    uint32_t device_ids[NUM_ROWS];
    float temperatures[NUM_ROWS];
    uint16_t status_codes[NUM_ROWS];

    for (int i = 0; i < NUM_ROWS; i++) {
        device_ids[i] = rows[i].device_id;
        temperatures[i] = rows[i].temperature;
        status_codes[i] = rows[i].status_code;
    }

    // 2. Query sizes for each column block
    // Type 2 = u32 (for device_ids)
    // Type 4 = float (for temperatures)
    // Type 1 = u16 (for status_codes)
    size_t size_ids = vbuf_pack_block(NULL, 201, 2, device_ids, NUM_ROWS);
    size_t size_temps = vbuf_pack_block(NULL, 202, 4, temperatures, NUM_ROWS);
    size_t size_status = vbuf_pack_block(NULL, 203, 1, status_codes, NUM_ROWS);

    printf("Required block sizes: ids=%zu, temps=%zu, status=%zu\n", size_ids, size_temps, size_status);

    // 3. Allocate combined buffer and pack all columns sequentially
    size_t global_header_size = 16;
    size_t total_buf_size = 1024; // Allocate plenty of space
    uint8_t *buffer = calloc(1, total_buf_size); // calloc to zero out memory
    assert(buffer != NULL);

    // Write the 16-byte global header manually:
    uint32_t *hdr32 = (uint32_t*)buffer;
    hdr32[0] = 0x46554256; // MAGIC
    hdr32[1] = 0x00050000; // VERSION
    buffer[8] = 4; // a_shift (1 << 4 = 16 alignment)
    hdr32[3] = total_buf_size; // DataLen

    // Pack each block directly into the buffer segments
    size_t offset = global_header_size;
    vbuf_pack_block(buffer + offset, 201, 2, device_ids, NUM_ROWS);
    offset += size_ids;
    offset = (offset + 15) & ~15; // Align next block to 16

    vbuf_pack_block(buffer + offset, 202, 4, temperatures, NUM_ROWS);
    offset += size_temps;
    offset = (offset + 15) & ~15; // Align next block to 16

    vbuf_pack_block(buffer + offset, 203, 1, status_codes, NUM_ROWS);

    // 4. Wrap packed buffer in a vbuf instance and query columns back
    // We mock the vbuf_instance_t
    vbuf_instance_t mock_inst;
    mock_inst.mem = buffer;
    mock_inst.size = total_buf_size;
    mock_inst.alignment = 16; // Use 16-byte alignment
    mock_inst.data = buffer + 16;

    size_t count_out;
    uint16_t width_out;

    // Read Device IDs (ID 201)
    const uint32_t *read_ids = (const uint32_t*)vbuf_get_col_ptr(&mock_inst, 201, &count_out, &width_out);
    assert(read_ids != NULL);
    assert(count_out == NUM_ROWS);
    assert(width_out == 32);

    // Read Temperatures (ID 202)
    const float *read_temps = (const float*)vbuf_get_col_ptr(&mock_inst, 202, &count_out, &width_out);
    assert(read_temps != NULL);
    assert(count_out == NUM_ROWS);
    assert(width_out == 32);

    // Read Status Codes (ID 203)
    const uint16_t *read_status = (const uint16_t*)vbuf_get_col_ptr(&mock_inst, 203, &count_out, &width_out);
    assert(read_status != NULL);
    assert(count_out == NUM_ROWS);
    assert(width_out == 16);

    // Validate retrieved data matches original row data
    for (int i = 0; i < NUM_ROWS; i++) {
        assert(read_ids[i] == rows[i].device_id);
        assert(read_temps[i] == rows[i].temperature);
        assert(read_status[i] == rows[i].status_code);
        printf("Row %d: ID=%u (read: %u), Temp=%.1f (read: %.1f), Status=%u (read: %u)\n",
               i, rows[i].device_id, read_ids[i],
               rows[i].temperature, read_temps[i],
               rows[i].status_code, read_status[i]);
    }

    free(buffer);
    printf("AoS-to-SoA Integration Test Passed successfully!\n\n");
}
