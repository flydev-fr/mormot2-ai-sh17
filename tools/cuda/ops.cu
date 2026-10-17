// LandrixAI - CUDA: the operations of a Qwen3 layer besides Linear (backlog B3,
// slice C3) - the CUDA forms of tools/vk/rmsnorm.comp, addnorm.comp,
// qkrope.comp, swiglu.comp and of the causal attention
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

// causal attention of token row r (grid y) and head h (grid x) over the
// cached keys 0 .. past + r: the scores into sc (cap per row and head), the
// softmax, then the values weighted - correct first (slice C5: tiles)
// - q and y: [rows, heads * headDim]; keys and values: [cap, kvHeads * headDim]
extern "C" __global__ void attention(const float* q, const float* keys,
  const float* values, float* y, float* sc, unsigned int heads,
  unsigned int headDim, unsigned int kvHeads, unsigned int past,
  unsigned int cap)
{
  __shared__ float part[4];
  unsigned int t = threadIdx.x;
  unsigned int h = blockIdx.x;
  unsigned int r = blockIdx.y;
  unsigned int d = headDim;
  unsigned int total = past + r + 1u; // causal: up to and including this row
  unsigned int g = h / (heads / kvHeads); // grouped-query: the shared KV head
  unsigned long long kvw = (unsigned long long)kvHeads * d;
  const float* qv = q + ((unsigned long long)r * heads + h) * d;
  float* s = sc + ((unsigned long long)r * heads + h) * cap;
  float scale = rsqrtf(float(d));
  float m = -3.402823466e38f;
  for (unsigned int j = t; j < total; j += BLOCK)
  {
    const float* kj = keys + j * kvw + (unsigned long long)g * d;
    float a = 0.0f;
    for (unsigned int c = 0u; c < d; c++)
      a += qv[c] * kj[c];
    a *= scale;
    s[j] = a;
    m = fmaxf(m, a);
  }
  m = blockMax(m, part);
  float sum = 0.0f;
  for (unsigned int j = t; j < total; j += BLOCK)
  {
    float e = expf(s[j] - m);
    s[j] = e;
    sum += e;
  }
  sum = blockSum(sum, part); // its __syncthreads also publishes s[]
  float inv = 1.0f / sum;
  float* yv = y + ((unsigned long long)r * heads + h) * d;
  for (unsigned int c = t; c < d; c += BLOCK)
  {
    const float* vc = values + (unsigned long long)g * d + c;
    float acc = 0.0f;
    for (unsigned int j = 0u; j < total; j++)
      acc += s[j] * vc[j * kvw];
    yv[c] = acc * inv;
  }
}
