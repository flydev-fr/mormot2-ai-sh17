// LandrixAI - CUDA kernels: element-wise basics (backlog B3, slice C1)
// - compiled to PTX by tools/cuda/build-kernels.sh, embedded as
//   PTX_BASIC in src/infer/mormot.ai.infer.cuda.ptx.inc
// - extern "C": the entry names stay unmangled for cuModuleGetFunction

// y[i] = a[i] + b[i] for i < n: one float per thread (the IEEE add is
// correctly rounded, so the bits match the CPU)
extern "C" __global__ void add(const float* a, const float* b, float* y,
  unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] + b[i];
}
