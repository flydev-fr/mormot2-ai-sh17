// LandrixAI - CUDA: y[r, o] = bias[o] + sum_i w[o, i] * x[r, i] (backlog B3,
// slice C2) - the CUDA form of tools/vk/linear.comp
// - w in its storage type: F32, Q8_0, Q4_K or Q6_K (ggml block layouts, see
//   mormot.ai.infer.tensor), dequantized per element in float, x in F32 - the
//   arithmetic of InferRowToF32 followed by an F32 dot product, in another
//   summation order (a tolerance, not bits)
// - one block of 128 threads per output feature (grid x) and token row
//   (grid y): correct first, the speed comes with slice C5
// - the half scales sit at even byte offsets in every block layout: a 16-bit
//   load is aligned; converted by cvt.f32.f16 (no cuda_fp16.h: it needs headers
//   the nvcc wheel does not carry)

// dtype: 1 = F32, 2 = Q8_0, 3 = Q4_K, 4 = Q6_K (TInferCudaBackend)

__device__ __forceinline__ float halfAt(const unsigned char* p)
{
  unsigned short h = *reinterpret_cast<const unsigned short*>(p);
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

// the weight of input i in the row at byte row
__device__ float weight(const unsigned char* row, unsigned int dtype, unsigned int i)
{
  if (dtype == 1u)
    return reinterpret_cast<const float*>(row)[i];
  if (dtype == 2u)
  {
    // Q8_0: 34 bytes per 32 values, value = d * qs[j]
    const unsigned char* b = row + (i >> 5) * 34u;
    return halfAt(b) * float(static_cast<signed char>(b[2u + (i & 31u)]));
  }
  if (dtype == 3u)
  {
    // Q4_K: 144 bytes per 256 values (InferScaleMinK4, DequantQ4K)
    const unsigned char* b = row + (i >> 8) * 144u;
    unsigned int k = i & 255u;
    unsigned int j = k >> 5;
    unsigned int sc, m;
    if (j < 4u)
    {
      sc = b[4u + j] & 63u;
      m = b[8u + j] & 63u;
    }
    else
    {
      sc = (b[8u + j] & 15u) | ((b[j] >> 6) << 4);
      m = (b[8u + j] >> 4) | ((b[4u + j] >> 6) << 4);
    }
    unsigned int q = b[16u + 32u * (j >> 1) + (k & 31u)];
    q = (j & 1u) ? q >> 4 : q & 15u;
    float d1 = halfAt(b) * float(sc);
    float m1 = halfAt(b + 2u) * float(m);
    return d1 * float(q) - m1;
  }
  // Q6_K: 210 bytes per 256 values - ql[128], qh[64], scales[16], d (Q6KValue)
  const unsigned char* b = row + (i >> 8) * 210u;
  unsigned int k = i & 255u;
  unsigned int n = k >> 7;
  unsigned int ii = k & 127u;
  unsigned int l = ii & 31u;
  unsigned int quarter = ii >> 5;
  unsigned int h = (b[128u + 32u * n + l] >> (2u * quarter)) & 3u;
  unsigned int ql = b[64u * n + l + ((quarter & 1u) ? 32u : 0u)];
  ql = quarter >= 2u ? ql >> 4 : ql & 15u;
  int q = int(ql | (h << 4)) - 32;
  float t = halfAt(b + 208u) * float(static_cast<signed char>(b[192u + (k >> 4)]));
  return t * float(q);
}

extern "C" __global__ void linear(const unsigned char* w, const float* x,
  const float* bias, float* y, unsigned int rows, unsigned int nin,
  unsigned int nout, unsigned int dtype, unsigned int hasBias)
{
  __shared__ float partial[4];
  unsigned int o = blockIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int t = threadIdx.x;
  // bytes of one weight row
  unsigned long long rowBytes;
  if (dtype == 1u)
    rowBytes = nin * 4ull;
  else if (dtype == 2u)
    rowBytes = (nin >> 5) * 34ull;
  else if (dtype == 3u)
    rowBytes = (nin >> 8) * 144ull;
  else
    rowBytes = (nin >> 8) * 210ull;
  const unsigned char* row = w + o * rowBytes;
  const float* xr = x + (unsigned long long)r * nin;
  float s = 0.0f;
  for (unsigned int i = t; i < nin; i += 128u)
    s += weight(row, dtype, i) * xr[i];
  // within the warp, then the four warps
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, off);
  if ((t & 31u) == 0u)
    partial[t >> 5] = s;
  __syncthreads();
  if (t == 0u)
  {
    float sum = partial[0] + partial[1] + partial[2] + partial[3];
    if (hasBias)
      sum += bias[o];
    y[(unsigned long long)r * nout + o] = sum;
  }
}
