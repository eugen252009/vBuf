# Tensor-Local CPU_REPACK Boundary

The pinned CPU_REPACK API supports a standalone valid backend buffer. Its
allocator delegates to the ordinary CPU buffer allocator, tensor initialization
stores the optimal repack trait in `tensor->extra`, and `set_tensor` invokes the
trait using only the tensor, source pointer, and tensor-local byte count. The
repack and specialized kernel paths do not inspect neighboring tensors or a
shared buffer-global table.

The tensor-local vBuf object reproduced the llama per-tensor state:
`12607488` bytes, offset zero, IQ4_NL descriptor, non-null
`iq4_nl_8x8_q8_0` trait, packed hash `3d8b9543543e6e57`, Q8_0 hash
`d2fd48314bc644ab`, `11756` bytes of CPU work plan, two threads, and
`n=10944,nr=1,nc=256,bs=2048` kernel geometry.

The specialized kernel executed and produced finite output, but the output
hash was `822669b42e16cde2`, not the established llama hash
`22960c20497db497`. Thus the shared 2.7-GB model-lifetime buffer is not needed
for tensor-local materialization, while CPU_REPACK storage/Q8/workspace
matching alone does not explain divergence #3.

The result separates two conclusions: tensor-local backend execution
materialization is valid, but bitwise llama compatibility remains unproven.
The remaining investigation must compare the actual pinned llama and vBuf
kernel implementation/path or obtain matching output vectors; no execution
buffer cache or persistent-format change is justified here.
