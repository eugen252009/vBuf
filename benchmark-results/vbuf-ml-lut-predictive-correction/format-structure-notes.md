# Canonical Format Structure

Q2_K: 16 scale/min bytes, 64 packed code bytes, and 4 FP16 super-scale/min bytes per 256-value block.

IQ2_XS: 2 FP16 scale bytes, 32 little-endian uint16 code units, and 8 scale bytes per 256-value block.

Q3_K: 32 high-bit bytes plus 64 low-bit code bytes are treated as code-bearing; 12 scale bytes and 2 FP16 scale bytes are copied unchanged.

IQ3_XXS: 2 FP16 scale bytes are copied unchanged and 96 code bytes are predicted.

The predictor never stores floating-point weights or numerical residuals. All metadata remains in the exact canonical payload.
