// LandrixAI - CUDA: the operations of a Qwen3 layer besides Linear (backlog B3,
// slice C3) - the CUDA forms of tools/vk/rmsnorm.comp, addnorm.comp,
// qkrope.comp, swiglu.comp, attention.comp and attnmerge.comp (slice C5)
// - float arithmetic as the Vulkan shaders: the logits match the CPU within a
//   tolerance, not in bits
// - every block reducing over a row has 128 threads (four warps)

#define BLOCK 128u

// the sum of v over the block (all 128 threads call it; the result in all)
__device__ float blockSum(float v, float* shared4)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  __syncthreads(); // shared4 may still be read from a previous call
  if ((threadIdx.x & 31u) == 0u)
    shared4[threadIdx.x >> 5] = v;
  __syncthreads();
  return shared4[0] + shared4[1] + shared4[2] + shared4[3];
}

// the maximum of v over the block
__device__ float blockMax(float v, float* shared4)
{
  for (unsigned int off = 16u; off > 0u; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  __syncthreads();
  if ((threadIdx.x & 31u) == 0u)
    shared4[threadIdx.x >> 5] = v;
  __syncthreads();
  return fmaxf(fmaxf(shared4[0], shared4[1]), fmaxf(shared4[2], shared4[3]));
}

// y[r] = w * x[r] / sqrt(mean(x[r]^2) + eps), one block per row; y may be x
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y,
  unsigned int cols, float eps)
{
  __shared__ float part[4];
  unsigned long long base = (unsigned long long)blockIdx.x * cols;
  float s = 0.0f;
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
  {
    float a = x[base + c];
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(cols) + eps);
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    y[base + c] = w[c] * (x[base + c] * inv);
}

// x[r] += t[r], then y[r] = w * x[r] / rms(x[r]): the residual and the next
// norm in one launch, one block per row
extern "C" __global__ void addnorm(float* x, const float* t, const float* w,
  float* y, unsigned int cols, float eps)
{
  __shared__ float part[4];
  unsigned long long base = (unsigned long long)blockIdx.x * cols;
  float s = 0.0f;
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
  {
    float a = x[base + c] + t[base + c];
    x[base + c] = a;
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(cols) + eps);
  for (unsigned int c = threadIdx.x; c < cols; c += BLOCK)
    y[base + c] = w[c] * (x[base + c] * inv);
}

// g = silu(g) * u over n values
extern "C" __global__ void swiglu(float* g, const float* u, unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float a = g[i];
    g[i] = (a / (1.0f + expf(-a))) * u[i];
  }
}

// per token row (grid y) and head (grid x: the heads, then the KV heads):
// RMSNorm of q (qn) or k (kn), rotary positions (cs: the cos of every row,
// then the sin), q in place, k and v stored into the cache at past + row
// - headDim <= 256, even
extern "C" __global__ void qkrope(float* q, const float* k, const float* v,
  const float* qn, const float* kn, const float* cs, float* kc, float* vc,
  unsigned int rows, unsigned int heads, unsigned int kvHeads,
  unsigned int headDim, unsigned int past, float eps)
{
  __shared__ float part[4];
  __shared__ float normed[256];
  unsigned int t = threadIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int hh = blockIdx.x;
  unsigned int d = headDim;
  unsigned int half = d >> 1;
  bool isQ = hh < heads;
  unsigned int h = isQ ? hh : hh - heads;
  unsigned long long src = isQ ? ((unsigned long long)r * heads + h) * d
                               : ((unsigned long long)r * kvHeads + h) * d;
  float s = 0.0f;
  for (unsigned int c = t; c < d; c += BLOCK)
  {
    float a = isQ ? q[src + c] : k[src + c];
    s += a * a;
  }
  float inv = rsqrtf(blockSum(s, part) / float(d) + eps);
  for (unsigned int c = t; c < d; c += BLOCK)
  {
    float a = isQ ? q[src + c] : k[src + c];
    normed[c] = (isQ ? qn[c] : kn[c]) * (a * inv);
  }
  __syncthreads();
  unsigned long long cbase = (unsigned long long)r * half;
  unsigned long long sbase = (unsigned long long)rows * half + cbase;
  unsigned long long kvw = (unsigned long long)kvHeads * d;
  unsigned long long dst = (past + (unsigned long long)r) * kvw + (unsigned long long)h * d;
  for (unsigned int i = t; i < half; i += BLOCK)
  {
    float c = cs[cbase + i];
    float sn = cs[sbase + i];
    float a = normed[i];
    float b = normed[i + half];
    float x0 = a * c - b * sn;
    float x1 = b * c + a * sn;
    if (isQ)
    {
      q[src + i] = x0;
      q[src + i + half] = x1;
    }
    else
    {
      kc[dst + i] = x0;
      kc[dst + i + half] = x1;
    }
  }
  if (!isQ)
    for (unsigned int c = t; c < d; c += BLOCK)
      vc[dst + c] = v[src + c];
}

// causal attention with grouped-query heads - the CUDA form of
// tools/vk/attention.comp (slice C5)
// - y[r, h] = softmax(q[r, h] . keys[0 .. past + r] / sqrt(d)) . values, the
//   keys and values of the KV head h / (heads / kvHeads), rows at kvWidth
// - one block of 128 per (head block: grid x, row block: grid y, split: grid
//   z); a block holds up to MAXQ queries - headBlock heads of one KV head times
//   rowBlock rows - so every key and value read serves all of them; the keys
//   go by in tiles of 128, their scores stay in shared memory and the softmax
//   is kept online (running maximum, sums rescaled)
// - the keys of a row are cut into splits ranges (flash-decoding), so a single
//   decoded token still fills the device; splits = 1: y is written normalized;
//   splits > 1: each range writes its unnormalized sums, its maximum and its
//   sum of exponentials to part[] ([heads, rows, splits, headDim + 2]), and
//   attnmerge combines them into y
// - headDim a multiple of 4 and at most 256 (the queries kept as float4, each
//   thread owns the output dimensions t and t + 128); keys 16-byte aligned
#define MAXQ 8
#define NEG -3.402823466e38f

extern "C" __global__ void attention(const float* q, const float4* keys4,
  const float* vals, float* y, float* part, unsigned int rows,
  unsigned int heads, unsigned int headDim, unsigned int kvHeads,
  unsigned int past, unsigned int splits, unsigned int headBlock,
  unsigned int rowBlock)
{
  __shared__ float4 qs[MAXQ * 64];   // the queries
  __shared__ float ps[MAXQ * 128];   // the scores, then the weights of a tile
  __shared__ float red[MAXQ * 128];  // the maximum of a tile
  unsigned int t = threadIdx.x;
  unsigned int hb = headBlock;
  unsigned int h0 = blockIdx.x * hb;
  unsigned int r0 = blockIdx.y * rowBlock;
  unsigned int s = blockIdx.z;
  unsigned int d = headDim;
  unsigned int d4 = d >> 2;
  unsigned long long kvw = (unsigned long long)kvHeads * d;
  unsigned long long qw = (unsigned long long)heads * d;
  unsigned int g = h0 / (heads / kvHeads);
  // query i: head h0 + i % hb, row r0 + i / hb
  unsigned int nq = hb * min(rowBlock, rows - r0);
  float scale = rsqrtf(float(d));
  for (unsigned int i = t; i < nq * d4; i += BLOCK)
  {
    unsigned int qi = i / d4;
    unsigned int k = i - qi * d4;
    unsigned long long qb = (r0 + qi / hb) * qw + (h0 + qi % hb) * d + 4u * k;
    qs[qi * 64u + k] = make_float4(q[qb], q[qb + 1u], q[qb + 2u], q[qb + 3u]);
  }
  // the keys j0 .. j1 - 1 of each query in this split (empty for a late
  // split of an early row); both grow with the row, the block reads lo .. hi
  unsigned int j0[MAXQ], j1[MAXQ];
  unsigned int lo = 0xffffffffu;
  unsigned int hi = 0u;
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
  {
    j0[i] = 0u;
    j1[i] = 0u;
    if (i < nq)
    {
      unsigned int total = past + r0 + i / hb + 1u;
      unsigned int chunk = (total + splits - 1u) / splits;
      j0[i] = min(total, s * chunk);
      j1[i] = min(total, j0[i] + chunk);
      lo = min(lo, j0[i]);
      hi = max(hi, j1[i]);
    }
  }
  __syncthreads();
  float m[MAXQ], l[MAXQ], acc0[MAXQ], acc1[MAXQ];
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
  {
    m[i] = NEG;
    l[i] = 0.0f;
    acc0[i] = 0.0f;
    acc1[i] = 0.0f;
  }
  unsigned long long vk = (unsigned long long)g * d + t;
  for (unsigned int base = lo; base < hi; base += BLOCK)
  {
    // the scores of key base + t
    unsigned int j = base + t;
    float sv[MAXQ];
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
      sv[i] = 0.0f;
    if (j < hi)
    {
      const float4* kp = keys4 + ((j * kvw + (unsigned long long)g * d) >> 2);
#pragma unroll 8
      for (unsigned int k = 0u; k < d4; k++)
      {
        float4 kv = kp[k];
#pragma unroll
        for (unsigned int i = 0u; i < MAXQ; i++)
          if (i < nq)
          {
            float4 qv = qs[i * 64u + k];
            sv[i] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
          }
      }
    }
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
    {
      float v = NEG;
      if ((i < nq) && (j >= j0[i]) && (j < j1[i]))
        v = sv[i] * scale;
      ps[i * 128u + t] = v;
      red[i * 128u + t] = v;
    }
    __syncthreads();
    for (unsigned int w = 64u; w > 0u; w >>= 1)
    {
      if (t < w)
#pragma unroll
        for (unsigned int i = 0u; i < MAXQ; i++)
          red[i * 128u + t] = fmaxf(red[i * 128u + t], red[i * 128u + t + w]);
      __syncthreads();
    }
    // the new maxima: the sums so far rescaled, the weights of this tile
#pragma unroll
    for (unsigned int i = 0u; i < MAXQ; i++)
    {
      float mn = fmaxf(m[i], red[i * 128u]);
      float f = expf(m[i] - mn);
      acc0[i] *= f;
      acc1[i] *= f;
      l[i] *= f;
      m[i] = mn;
      float e = 0.0f;
      if ((i < nq) && (j >= j0[i]) && (j < j1[i]))
        e = expf(ps[i * 128u + t] - mn);
      ps[i * 128u + t] = e;
    }
    __syncthreads();
    // the weighted values of this tile
    unsigned int n = min(BLOCK, hi - base);
#pragma unroll 8
    for (unsigned int jj = 0u; jj < n; jj++)
    {
      unsigned long long vb = (base + jj) * kvw + vk;
      float v0 = 0.0f;
      float v1 = 0.0f;
      if (t < d)
        v0 = vals[vb];
      if (t + 128u < d)
        v1 = vals[vb + 128u];
#pragma unroll
      for (unsigned int i = 0u; i < MAXQ; i++)
      {
        float w = ps[i * 128u + jj];
        l[i] += w;
        acc0[i] += w * v0;
        acc1[i] += w * v1;
      }
    }
    __syncthreads(); // ps and red are refilled by the next tile
  }
#pragma unroll
  for (unsigned int i = 0u; i < MAXQ; i++)
    if (i < nq)
    {
      unsigned int h = h0 + i % hb;
      unsigned int r = r0 + i / hb;
      if (splits == 1u)
      {
        unsigned long long yb = r * qw + (unsigned long long)h * d + t;
        if (t < d)
          y[yb] = acc0[i] / l[i];
        if (t + 128u < d)
          y[yb + 128u] = acc1[i] / l[i];
      }
      else
      {
        unsigned long long pb = (((unsigned long long)h * rows + r) * splits + s) * (d + 2u);
        if (t < d)
          part[pb + t] = acc0[i];
        if (t + 128u < d)
          part[pb + t + 128u] = acc1[i];
        if (t == 0u)
        {
          part[pb + d] = m[i];
          part[pb + d + 1u] = l[i];
        }
      }
    }
}

// combines the key ranges of attention (flash-decoding), the CUDA form of
// tools/vk/attnmerge.comp: per (head: grid x, row: grid y) M = max of the
// range maxima, y = sum_s exp(m_s - M) * acc_s / sum_s exp(m_s - M) * l_s
// - an empty range has m = -FLT_MAX and l = 0: its weight is 0
extern "C" __global__ void attnmerge(const float* part, float* y,
  unsigned int rows, unsigned int heads, unsigned int headDim,
  unsigned int splits)
{
  unsigned int t = threadIdx.x;
  unsigned int h = blockIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int d = headDim;
  unsigned int stride = d + 2u;
  const float* b = part + ((unsigned long long)h * rows + r) * splits * stride;
  float mx = NEG;
  for (unsigned int s = 0u; s < splits; s++)
    mx = fmaxf(mx, b[s * stride + d]);
  float den = 0.0f;
  for (unsigned int s = 0u; s < splits; s++)
    den += expf(b[s * stride + d] - mx) * b[s * stride + d + 1u];
  for (unsigned int k = t; k < d; k += BLOCK)
  {
    float acc = 0.0f;
    for (unsigned int s = 0u; s < splits; s++)
      acc += expf(b[s * stride + d] - mx) * b[s * stride + k];
    y[((unsigned long long)r * heads + h) * d + k] = acc / den;
  }
}
