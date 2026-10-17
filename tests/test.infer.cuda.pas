/// tests of the native inference engine: the CUDA backend (backlog B3)
// - skips without an NVIDIA driver and device; LAI_REQUIRE_CUDA=1 turns
//   the skip into a failure (a run that must cover the GPU)
unit test.infer.cuda;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.cuda,
  mormot.ai.infer.backend.cuda,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.qwen3.cuda,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  mormot.ai.infer.sampler,
  test.infer.ref,
  test.infer.fast,
  test.infer.bpe;

type
  /// slice C1: driver, device, modules from PTX, memory and a kernel
  TTestInferCuda = class(TSynTestCase)
  protected
    fCu: TInferCuda;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) without a device
    function HasDevice: boolean;
  published
    /// y = a + b on the device: the bits of the CPU sum (an IEEE add is
    // correctly rounded on both), over a count no multiple of the block,
    // twice with the same kernel and other buffers
    procedure AddMatchesCpu;
    /// misuse is refused: by the checks before the driver, or by the driver
    // as an EInferCuda with its error name
    procedure RefusesMisuse;
    /// device memory: what goes up comes down, at an offset, also in pieces
    procedure UploadDownload;
    /// slice C2: TInferCudaBackend.Linear for F32, Q8_0, Q4_K and Q6_K weights
    // against the reference's F32 product of the dequantized weights (a
    // tolerance: another summation order), with and without bias, more than
    // 65535 outputs, the weight cache, and the CPU for other types
    procedure LinearMatchesReference;
    /// slice C3 (attention: C5): rmsnorm, addnorm, swiglu, qkrope and attention of
    // tools/cuda/ops.cu against the CPU reference's RmsNorm, Add, SwiGlu, Rope
    // and CausalAttention (a tolerance: float arithmetic, another order),
    // enqueued without a wait between them
    procedure OpsMatchCpu;
    /// slice C4: the tiny Qwen3 forward on the device against the CPU
    // reference - every row of a prompt longer than one chunk, decoding, a
    // reused prefix, the weights uploaded again; F32 from safetensors
    // (relative error), Q8_0 from GGUF (cosine: the CPU quantizes the
    // activations, the device does not)
    procedure TinyQwen3MatchesCpu;
    /// the tiny Qwen3 with a head dimension of 256 (the widest qkrope takes)
    procedure TinyHeadDim256MatchesCpu;
    /// Qwen3-0.6B (Q4_K_M and Q8_0 GGUF) on the device against the fast CPU
    // backend, teacher-forced - skipped without the model files
    procedure Qwen3GgufMatchesCpu;
    /// TNativeLlmClient on the device (Device := TQwen3Cuda): the greedy
    // answer of Qwen3-0.6B Q8_0 as without it, the next turn from the prompt
    // cache on the device, back to the backend with Device := nil
    procedure NativeClientOnDevice;
  end;


implementation

/// deterministic pseudo-random values in [-aScale, aScale)
procedure Fill(var aValues: TSingleDynArray; aCount: PtrInt; aSeed: cardinal;
  aScale: single);
var
  i: PtrInt;
begin
  SetLength(aValues, aCount);
  for i := 0 to aCount - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    aValues[i] := ((aSeed shr 8) / 16777216 - 0.5) * 2 * aScale;
  end;
end;

procedure TTestInferCuda.Setup;
begin
  if TInferCuda.Available then
  begin
    fCu := TInferCuda.Create;
    AddConsole('CUDA device: % (compute %.%, % SMs, % MB, driver %)',
      [fCu.DeviceName, fCu.ComputeMajor, fCu.ComputeMinor, fCu.Multiprocessors,
       fCu.TotalMemory shr 20, fCu.DriverVersion]);
  end;
end;

procedure TTestInferCuda.CleanUp;
begin
  FreeAndNil(fCu);
end;

function TTestInferCuda.HasDevice: boolean;
var
  refused: boolean;
begin
  result := fCu <> nil;
  if result then
    exit;
  // without a device a direct Create raises EInferCuda - nothing else (no
  // release of a context it never took)
  refused := false;
  try
    TInferCuda.Create.Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'TInferCuda.Create without a device raises EInferCuda');
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_CUDA') <> '1',
    'LAI_REQUIRE_CUDA=1 but no CUDA device (%)', [CUDA_LIBRARY]);
  {$ifdef CPU64}
  AddConsole('SKIP CUDA: no driver or no device (%)', [CUDA_LIBRARY]);
  {$else}
  AddConsole('SKIP CUDA: a 64-bit process is needed');
  {$endif CPU64}
end;

procedure TTestInferCuda.AddMatchesCpu;
const
  COUNTS: array[0..1] of cardinal = (1000003, 17);
  BLOCK = 256;
var
  module: TInferCuModule;
  kernel: TCuHandle;
  a, b, y: TInferCuBuffer;
  va, vb, vy: TSingleDynArray;
  i, k, bad: PtrInt;
  n: cardinal; // the kernel's unsigned int
  pa, pb, py: TCuDevicePtr;
  sum: single;
begin
  if not HasDevice then
    exit;
  module := TInferCuModule.Create(fCu, @PTX_BASIC);
  try
    kernel := module.Kernel('add');
    for k := 0 to high(COUNTS) do
    begin
      n := COUNTS[k];
      SetLength(va, n);
      SetLength(vb, n);
      SetLength(vy, n);
      for i := 0 to n - 1 do
      begin
        va[i] := (i mod 1000) * 0.001 - 0.37;
        vb[i] := 1 / (i + 3);
      end;
      a := nil;
      b := nil;
      y := nil;
      try
        a := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        b := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        y := TInferCuBuffer.Create(fCu, n * SizeOf(single));
        a.Upload(0, pointer(va), n * SizeOf(single));
        b.Upload(0, pointer(vb), n * SizeOf(single));
        pa := a.Ptr;
        pb := b.Ptr;
        py := y.Ptr;
        fCu.Launch(kernel, [@pa, @pb, @py, @n], (n + BLOCK - 1) div BLOCK, BLOCK);
        y.Download(0, pointer(vy), n * SizeOf(single));
      finally
        y.Free;
        b.Free;
        a.Free;
      end;
      bad := 0;
      for i := 0 to n - 1 do
      begin
        sum := va[i] + vb[i];
        if PCardinal(@sum)^ <> PCardinal(@vy[i])^ then
          inc(bad);
      end;
      CheckUtf8(bad = 0, 'add of %: % of the sums differ from the CPU', [n, bad]);
    end;
  finally
    module.Free;
  end;
end;

procedure TTestInferCuda.RefusesMisuse;
var
  module: TInferCuModule;
  buf: TInferCuBuffer;
  tmp: array[0..15] of byte;
  refused: boolean;
  msg: string;
begin
  if not HasDevice then
    exit;
  refused := false;
  try
    TInferCuBuffer.Create(fCu, 0).Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'a buffer of no bytes');
  buf := TInferCuBuffer.Create(fCu, 16);
  try
    refused := false;
    try
      buf.Upload(8, @tmp, 16);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'an upload past the end');
    refused := false;
    try
      buf.Download(-1, @tmp, 4);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'a download before the start');
    msg := '';
    try
      // offset + size beyond Int64: the range check must not wrap around and
      // let the copy reach the driver
      buf.Upload(8, @tmp, high(Int64) - 4);
    except
      on e: EInferCuda do
        msg := e.Message;
    end;
    Check(Pos('.Upload:', msg) > 0, 'an upload whose end overflows: ' + msg);
  finally
    buf.Free;
  end;
  module := TInferCuModule.Create(fCu, @PTX_BASIC);
  try
    msg := '';
    try
      module.Kernel('no_such_kernel');
    except
      on e: EInferCuda do
        msg := e.Message;
    end;
    // the driver's error name comes with the call that failed
    Check(Pos('CUDA_ERROR_NOT_FOUND', msg) > 0, msg);
    refused := false;
    try
      fCu.Launch(module.Kernel('add'), [], 0, 256);
    except
      on EInferCuda do
        refused := true;
    end;
    Check(refused, 'an empty grid');
  finally
    module.Free;
  end;
  refused := false;
  try
    // not PTX: the driver refuses to load it
    TInferCuModule.Create(fCu, PAnsiChar('not a module')).Free;
  except
    on EInferCuda do
      refused := true;
  end;
  Check(refused, 'a module that is no PTX');
end;

procedure TTestInferCuda.UploadDownload;
const
  N = 4096;
var
  buf, other: TInferCuBuffer;
  up, down, zero: array[0..N - 1] of byte;
  i: PtrInt;
  refused: boolean;
begin
  if not HasDevice then
    exit;
  for i := 0 to N - 1 do
    up[i] := byte(i * 7 + 3);
  FillCharFast(zero, SizeOf(zero), 0);
  buf := TInferCuBuffer.Create(fCu, N);
  try
    Check(buf.Size = N, 'size');
    // in three pieces, one at an odd offset
    buf.Upload(0, @up[0], 1000);
    buf.Upload(1000, @up[1000], 1);
    buf.Upload(1001, @up[1001], N - 1001);
    FillCharFast(down, SizeOf(down), 0);
    buf.Download(0, @down, N);
    Check(CompareMem(@up, @down, N), 'whole');
    FillCharFast(down, SizeOf(down), 0);
    buf.Download(777, @down, 333);
    Check(CompareMem(@up[777], @down, 333), 'at an offset');
    // device to device, between offsets, then back
    other := TInferCuBuffer.Create(fCu, N);
    try
      other.Upload(0, @zero, N);
      buf.CopyTo(100, other, 3001, 1000);
      FillCharFast(down, SizeOf(down), 0);
      other.Download(0, @down, N);
      Check(CompareMem(@up[100], @down[3001], 1000), 'CopyTo: the bytes');
      Check(CompareMem(@zero, @down, 3001) and
        CompareMem(@zero, @down[4001], N - 4001), 'CopyTo: nothing else');
      refused := false;
      try
        buf.CopyTo(0, other, 3500, 1000); // past the end of other
      except
        on EInferCuda do
          refused := true;
      end;
      Check(refused, 'CopyTo past the end');
    finally
      other.Free;
    end;
  finally
    buf.Free;
  end;
end;


procedure TTestInferCuda.LinearMatchesReference;
const
  // rows, in, out: one row, partial tiles, more than 65535 outputs
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 256, 1), (3, 512, 17), (37, 1024, 33), (5, 3072, 20), (1, 256, 70001),
    (70, 512, 130));
  TYPES: array[0..3] of TInferDType = (idtF32, idtQ8_0, idtQ4_K, idtQ6_K);
var
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  fast: TInferCpuFastBackend;
  t, i, o, rows, nin, nout: PtrInt;
  kind: TInferDType;
  x, bias, wf, wd, y1, y2, y3: TSingleDynArray;
  wq: RawByteString;
  data: pointer;
  b: PSingleArray;
  e: double;
  calls: Int64;
begin
  if not HasDevice then
    exit;
  gpu := TInferCudaBackend.Create;
  ref := TInferCpuBackend.Create;
  fast := TInferCpuFastBackend.Create;
  try
    for t := 0 to high(TYPES) do
    begin
      kind := TYPES[t];
      for i := 0 to high(SHAPES) do
      begin
        rows := SHAPES[i, 0];
        nin := SHAPES[i, 1];
        nout := SHAPES[i, 2];
        Fill(x, rows * nin, 11 + i, 3);
        Fill(bias, nout, 21 + i, 1);
        b := nil;
        if not odd(i) then
          b := pointer(bias);
        case kind of
          idtF32:
            begin
              Fill(wf, nout * nin, 31 + i, 0.1);
              data := pointer(wf);
            end;
          idtQ8_0:
            begin
              wq := RandomQ8(nout, nin, 41 + i);
              data := pointer(wq);
            end;
        else
          begin
            wq := RandomK(kind, nout, nin, 51 + i + 10 * t);
            data := pointer(wq);
          end;
        end;
        SetLength(wd, nout * nin);
        for o := 0 to nout - 1 do
          InferRowToF32(data, kind, o, nin, @wd[o * nin]);
        SetLength(y1, rows * nout);
        SetLength(y2, rows * nout);
        ref.Linear(pointer(x), rows, nin, InferWeight(pointer(wd), idtF32), nout,
          b, pointer(y1));
        // the weights of the previous shape are freed: their address may come
        // back for other data of the same size (the cache is keyed by it)
        gpu.ClearWeights;
        calls := gpu.GpuLinears;
        gpu.Linear(pointer(x), rows, nin, InferWeight(data, kind), nout, b,
          pointer(y2));
        CheckUtf8(gpu.GpuLinears = calls + 1, '% on the device', [INFER_DTYPE_NAME[kind]]);
        e := RelError(pointer(y2), pointer(y1), rows * nout);
        CheckUtf8(e <= 1e-5, '% % x % x %: relative error %',
          [INFER_DTYPE_NAME[kind], rows, nin, nout, e]);
        // again, from the weight cache: the same bits
        SetLength(y3, rows * nout);
        gpu.Linear(pointer(x), rows, nin, InferWeight(data, kind), nout, b,
          pointer(y3));
        CheckUtf8(SameBits(pointer(y2), pointer(y3), rows * nout),
          '% cached weights', [INFER_DTYPE_NAME[kind]]);
      end;
    end;
    // a weight type without a kernel: the CPU, bit for bit as the fast backend
    wq := RandomQ4(8, 64, 5);
    Fill(x, 2 * 64, 3, 2);
    SetLength(y1, 2 * 8);
    SetLength(y2, 2 * 8);
    calls := gpu.GpuLinears;
    gpu.Linear(pointer(x), 2, 64, InferWeight(pointer(wq), idtQ4_0), 8, nil,
      pointer(y1));
    fast.Linear(pointer(x), 2, 64, InferWeight(pointer(wq), idtQ4_0), 8, nil,
      pointer(y2));
    Check(gpu.GpuLinears = calls, 'Q4_0 stays on the CPU');
    Check(SameBits(pointer(y1), pointer(y2), 2 * 8), 'Q4_0 as the fast backend');
  finally
    fast.Free;
    ref.Free;
    gpu.Free;
  end;
end;


procedure TTestInferCuda.OpsMatchCpu;
var
  ops: TInferCuModule;
  ref: TInferCpuBackend;
  kNorm, kAdd, kSwi, kQk, kAtt, kMerge: TCuHandle;
  bufs: array of TInferCuBuffer;

  // a device copy of v (freed at the end)
  function Up(const v: TSingleDynArray): TCuDevicePtr;
  var
    b: TInferCuBuffer;
  begin
    b := TInferCuBuffer.Create(fCu, length(v) * 4 + 4);
    SetLength(bufs, length(bufs) + 1);
    bufs[high(bufs)] := b;
    if v <> nil then
      b.Upload(0, pointer(v), length(v) * 4);
    result := b.Ptr;
  end;

  // the buffer of aPtr back into v (its length), after every enqueued kernel
  procedure Down(aPtr: TCuDevicePtr; var v: TSingleDynArray);
  var
    i: PtrInt;
  begin
    fCu.Synchronize;
    for i := 0 to high(bufs) do
      if bufs[i].Ptr = aPtr then
      begin
        bufs[i].Download(0, pointer(v), length(v) * 4);
        exit;
      end;
  end;

  procedure Near(const a, b: TSingleDynArray; aTol: double; const aWhat: RawUtf8);
  var
    e: double;
  begin
    e := RelError(pointer(a), pointer(b), length(b));
    CheckUtf8(e <= aTol, '%: relative error %', [aWhat, e]);
  end;

  procedure NormAndAdd(aRows, aCols: PtrInt);
  var
    x, t, w, y, xc, yc: TSingleDynArray;
    px, pt, pw, py: TCuDevicePtr;
    cols: cardinal;
    eps: single;
  begin
    Fill(x, aRows * aCols, 7 + aCols, 3);
    Fill(t, aRows * aCols, 8 + aCols, 2);
    Fill(w, aCols, 9 + aCols, 1);
    SetLength(y, aRows * aCols);
    SetLength(yc, aRows * aCols);
    cols := aCols;
    eps := 1e-6;
    // rmsnorm
    px := Up(x);
    pw := Up(w);
    py := Up(y);
    fCu.Enqueue(kNorm, [@px, @pw, @py, @cols, @eps], aRows, 128);
    Down(py, yc);
    ref.RmsNorm(pointer(x), aRows, aCols, pointer(w), eps, pointer(y));
    Near(yc, y, 1e-5, FormatUtf8('rmsnorm % x %', [aRows, aCols]));
    // addnorm: x += t, then the norm of x
    pt := Up(t);
    fCu.Enqueue(kAdd, [@px, @pt, @pw, @py, @cols, @eps], aRows, 128);
    SetLength(xc, aRows * aCols);
    Down(px, xc);
    Down(py, yc);
    ref.Add(pointer(x), pointer(t), aRows * aCols);
    ref.RmsNorm(pointer(x), aRows, aCols, pointer(w), eps, pointer(y));
    Near(xc, x, 1e-6, FormatUtf8('addnorm residual % x %', [aRows, aCols]));
    Near(yc, y, 1e-5, FormatUtf8('addnorm % x %', [aRows, aCols]));
  end;

  procedure SwiGluCase(aCount: PtrInt);
  var
    g, u, gc: TSingleDynArray;
    pg, pu: TCuDevicePtr;
    n: cardinal;
  begin
    Fill(g, aCount, 21, 6);
    Fill(u, aCount, 22, 2);
    pg := Up(g);
    pu := Up(u);
    n := aCount;
    fCu.Enqueue(kSwi, [@pg, @pu, @n], (n + 127) div 128, 128);
    SetLength(gc, aCount);
    Down(pg, gc);
    ref.SwiGlu(pointer(g), pointer(u), aCount);
    Near(gc, g, 1e-5, FormatUtf8('swiglu %', [aCount]));
  end;

  // qkrope of aRows new rows after aPast cached ones, then the attention of
  // those rows over the whole cache - both against the CPU; the attention
  // blocks hold aHeadBlock heads times 8 div aHeadBlock rows, the keys of a
  // row cut into aSplits ranges (merged by attnmerge when more than one)
  procedure AttentionCase(aRows, aHeads, aKvHeads, aHeadDim, aPast, aHeadBlock,
    aSplits: PtrInt);
  var
    q, k, v, qn, kn, cs, cosv, sinv, keys, values, y, kc, vc, qc, yc,
      part: TSingleDynArray;
    pq, pk, pv, pqn, pkn, pcs, pkc, pvc, py, ppart: TCuDevicePtr;
    rows, heads, kvHeads, headDim, past, splits, hb, rb: cardinal;
    eps: single;
    half, kvw, cp, r, i: PtrInt;
  begin
    half := aHeadDim shr 1;
    kvw := aKvHeads * aHeadDim;
    cp := aPast + aRows + 3; // a cache larger than used
    Fill(q, aRows * aHeads * aHeadDim, 31, 2);
    Fill(k, aRows * kvw, 32, 2);
    Fill(v, aRows * kvw, 33, 2);
    Fill(qn, aHeadDim, 34, 1);
    Fill(kn, aHeadDim, 35, 1);
    // rotary factors of the new rows: cos, then sin (as TQwen3Model.Rotary)
    SetLength(cosv, aRows * half);
    SetLength(sinv, aRows * half);
    for r := 0 to aRows - 1 do
      for i := 0 to half - 1 do
      begin
        cosv[r * half + i] := Cos((aPast + r) * (i + 1) * 0.01);
        sinv[r * half + i] := Sin((aPast + r) * (i + 1) * 0.01);
      end;
    cs := copy(cosv);
    SetLength(cs, 2 * aRows * half);
    MoveFast(pointer(sinv)^, cs[aRows * half], aRows * half * 4);
    // the cached rows: random keys and values, the new ones written by qkrope
    Fill(keys, cp * kvw, 36, 1);
    Fill(values, cp * kvw, 37, 1);
    pq := Up(q);
    pk := Up(k);
    pv := Up(v);
    pqn := Up(qn);
    pkn := Up(kn);
    pcs := Up(cs);
    pkc := Up(keys);
    pvc := Up(values);
    rows := aRows;
    heads := aHeads;
    kvHeads := aKvHeads;
    headDim := aHeadDim;
    past := aPast;
    splits := aSplits;
    hb := aHeadBlock;
    rb := 8 div aHeadBlock;
    eps := 1e-6;
    // every buffer up first: the two kernels then run back to back, with no
    // copy (which waits for the stream) between them
    SetLength(y, aRows * aHeads * aHeadDim);
    py := Up(y);
    SetLength(part, aHeads * aRows * aSplits * (aHeadDim + 2));
    ppart := Up(part);
    fCu.Enqueue(kQk, [@pq, @pk, @pv, @pqn, @pkn, @pcs, @pkc, @pvc, @rows, @heads,
      @kvHeads, @headDim, @past, @eps], aHeads + aKvHeads, 128, aRows);
    fCu.Enqueue(kAtt, [@pq, @pkc, @pvc, @py, @ppart, @rows, @heads, @headDim,
      @kvHeads, @past, @splits, @hb, @rb], aHeads div aHeadBlock, 128,
      (aRows + rb - 1) div rb, aSplits);
    if aSplits > 1 then
      fCu.Enqueue(kMerge, [@ppart, @py, @rows, @heads, @headDim, @splits],
        aHeads, 128, aRows);
    SetLength(qc, length(q));
    SetLength(kc, length(keys));
    SetLength(vc, length(values));
    SetLength(yc, length(y));
    Down(pq, qc);
    Down(pkc, kc);
    Down(pvc, vc);
    Down(py, yc);
    // the CPU: per-head norms, rotary positions, the store into the cache
    ref.RmsNorm(pointer(q), aRows * aHeads, aHeadDim, pointer(qn), eps, pointer(q));
    ref.RmsNorm(pointer(k), aRows * aKvHeads, aHeadDim, pointer(kn), eps, pointer(k));
    ref.Rope(pointer(q), aRows, aHeads, aHeadDim, pointer(cosv), pointer(sinv));
    ref.Rope(pointer(k), aRows, aKvHeads, aHeadDim, pointer(cosv), pointer(sinv));
    MoveFast(pointer(k)^, keys[aPast * kvw], aRows * kvw * 4);
    MoveFast(pointer(v)^, values[aPast * kvw], aRows * kvw * 4);
    Near(qc, q, 1e-5, FormatUtf8('qkrope q % rows, % heads of %',
      [aRows, aHeads, aHeadDim]));
    Near(kc, keys, 1e-5, FormatUtf8('qkrope keys after %', [aPast]));
    Check(CompareMem(pointer(vc), pointer(values), length(values) * 4),
      'qkrope values copied, the rest of the cache untouched');
    ref.CausalAttention(pointer(q), aRows, aHeads, aHeadDim, pointer(keys),
      pointer(values), aPast, aKvHeads, pointer(y));
    Near(yc, y, 2e-5, FormatUtf8(
      'attention % rows after %, % heads / % KV heads of %, % per block, % splits',
      [aRows, aPast, aHeads, aKvHeads, aHeadDim, aHeadBlock, aSplits]));
  end;

var
  i: PtrInt;
begin
  if not HasDevice then
    exit;
  ref := TInferCpuBackend.Create;
  ops := TInferCuModule.Create(fCu, @PTX_OPS);
  try
    kNorm := ops.Kernel('rmsnorm');
    kAdd := ops.Kernel('addnorm');
    kSwi := ops.Kernel('swiglu');
    kQk := ops.Kernel('qkrope');
    kAtt := ops.Kernel('attention');
    kMerge := ops.Kernel('attnmerge');
    NormAndAdd(5, 1024);
    NormAndAdd(3, 100); // no multiple of the block
    SwiGluCase(100003);
    AttentionCase(7, 4, 2, 64, 0, 2, 1);     // a prompt, 4 rows per block
    AttentionCase(7, 4, 2, 64, 0, 1, 3);     // early rows: empty late ranges
    AttentionCase(3, 16, 8, 128, 20, 2, 1);  // more rows after a cached prefix
    AttentionCase(1, 16, 8, 128, 300, 2, 1); // decoding: more keys than a tile
    AttentionCase(1, 16, 8, 128, 300, 2, 5); // decoding, the keys in 5 ranges
    AttentionCase(2, 4, 4, 256, 5, 1, 1);    // the widest head
    AttentionCase(2, 4, 4, 256, 5, 1, 2);
    AttentionCase(9, 8, 1, 64, 130, 8, 4);   // 8 heads of one KV head
  finally
    for i := 0 to high(bufs) do
      bufs[i].Free;
    ops.Free;
    ref.Free;
  end;
end;


procedure TTestInferCuda.TinyQwen3MatchesCpu;
const
  PROMPT = 300; // more than one chunk of the device forward pass (256)
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2: TSingleDynArray;
  i, k, v: PtrInt;
  seed: cardinal;
  e: double;
  cmp: TInferCompare;
  id: integer;
  out1, out2: TIntegerDynArray;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 7;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  for k := 0 to 1 do
  begin
    gpu := TInferCudaBackend.Create;
    ref := TInferCpuBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      if k = 0 then
        model := TQwen3Model.Create(dir, ref)
      else
        model := TQwen3Model.CreateFromGguf(IncludeTrailingPathDelimiter(FindInferRefDir('tiny-qwen3-gguf')) +
          'q8_0.gguf', ref);
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      c1 := model.NewContext;
      c2 := vk.NewContext;
      // the prompt: every row's logits
      SetLength(l1, PROMPT * v);
      SetLength(l2, PROMPT * v);
      model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
      vk.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
      Check(c2.Length = PROMPT, 'context length');
      if k = 0 then
      begin
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= 1e-4, 'F32 prompt logits: relative error %', [e]);
      end
      else
      begin
        cmp := InferCompare(pointer(l2), pointer(l1), PROMPT * v, 0.05, 0.02);
        CheckUtf8(cmp.Cosine >= 0.999, 'Q8_0 prompt logits: %',
          [InferCompareText(cmp)]);
      end;
      // decoding, one token at a time - the keys of a row in one range or
      // cut into ranges of 64
      for i := 0 to 4 do
      begin
        id := ids[i];
        vk.SplitKeys := (i and 1) * 64;
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
        if k = 0 then
        begin
          e := RelError(pointer(l2), pointer(l1), v);
          CheckUtf8(e <= 1e-4, 'F32 decode step %: relative error %', [i, e]);
        end
        else
        begin
          cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
          CheckUtf8(cmp.Cosine >= 0.999, 'Q8_0 decode step %: %',
            [i, InferCompareText(cmp)]);
        end;
      end;
      // a reused prefix: the cache cut back, then other tokens
      Check(vk.Reuse(c2, pointer(ids), 100) = 99, 'reuse');
      c1.Truncate(99);
      model.Forward(c1, @ids[150], 20, pointer(l1));
      vk.Forward(c2, @ids[150], 20, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after reuse: %', [InferCompareText(cmp)]);
      // the backend forgot its weights: the next forward uploads them again
      gpu.ClearWeights;
      id := ids[200];
      model.Forward(c1, @id, 1, pointer(l1));
      vk.Forward(c2, @id, 1, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after ClearWeights: %', [InferCompareText(cmp)]);
      // greedy: every returned token is cached, as on the CPU
      c1.Reset;
      c2.Reset;
      out1 := model.Greedy(c1, copy(ids, 0, 10), 6, []);
      out2 := vk.Greedy(c2, copy(ids, 0, 10), 6, []);
      Check((length(out2) = 6) and (c2.Length = 10 + 6), 'greedy cached its tokens');
      Check((length(out1) = 6) and (c1.Length = c2.Length), 'greedy lengths');
      // a fresh context (64 positions) grows with 10 tokens cached: they are
      // copied into the larger cache
      FreeAndNil(c2);
      c2 := vk.NewContext;
      c1.Reset;
      model.Forward(c1, pointer(ids), 10, pointer(l1));
      vk.Forward(c2, pointer(ids), 10, pointer(l2));
      model.Forward(c1, @ids[10], 100, pointer(l1));
      vk.Forward(c2, @ids[10], 100, pointer(l2));
      cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
      CheckUtf8(cmp.Cosine >= 0.999, 'after the cache grew: %', [InferCompareText(cmp)]);
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      ref.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.Qwen3GgufMatchesCpu;
const
  FILES: array[0..1] of TFileName = ('qwen3-0.6b-q4_k_m.gguf', 'qwen3-0.6b-q8_0.gguf');
  // "Wie verlege ich ein Kupferrohr?" - any ids do; these are ordinary words
  PROMPT: array[0..11] of integer = (
    151644, 872, 198, 54, 645, 2739, 273, 10109, 12153, 3105, 63084, 30);
  STEPS = 16;
var
  dir, gguf: TFileName;
  f, i, v, agree: PtrInt;
  gpu: TInferCudaBackend;
  cpu: TInferCpuFastBackend;
  model: TQwen3Model;
  vk: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  l1, l2: TSingleDynArray;
  cmp: TInferCompare;
  worst: double;
  id: integer;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-0.6B on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  dir := FindQwen3Folder('model.safetensors');
  for f := 0 to high(FILES) do
  begin
    gguf := '';
    if dir <> '' then
      gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
        FILES[f]);
    if (gguf = '') or not FileExists(gguf) then
    begin
      CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
        'LAI_REQUIRE_MODEL=1 but no vendor/models/%', [FILES[f]]);
      AddConsole('SKIP Qwen3-0.6B on the device: no vendor/models/%', [FILES[f]]);
      continue;
    end;
    gpu := TInferCudaBackend.Create;
    cpu := TInferCpuFastBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      model := TQwen3Model.CreateFromGguf(gguf, cpu);
      v := model.Config.VocabSize;
      vk := TQwen3Cuda.Create(model, gpu);
      c1 := model.NewContext;
      c2 := vk.NewContext;
      SetLength(l1, v);
      SetLength(l2, v);
      model.Forward(c1, @PROMPT[0], length(PROMPT), pointer(l1));
      vk.Forward(c2, @PROMPT[0], length(PROMPT), pointer(l2));
      worst := 1;
      agree := 0;
      for i := 0 to STEPS - 1 do
      begin
        cmp := InferCompare(pointer(l2), pointer(l1), v, 0.05, 0.02);
        if cmp.Cosine < worst then
          worst := cmp.Cosine;
        id := InferArgMax(pointer(l1), v);
        if InferArgMax(pointer(l2), v) = id then
          inc(agree);
        // the CPU's token to both: the comparison stays on one path
        model.Forward(c1, @id, 1, pointer(l1));
        vk.Forward(c2, @id, 1, pointer(l2));
      end;
      AddConsole('% on %: lowest logits cosine %, argmax agrees % of %',
        [FILES[f], fCu.DeviceName, worst, agree, STEPS]);
      CheckUtf8(worst >= 0.99, '% logits cosine %', [FILES[f], worst]);
      CheckUtf8(agree >= STEPS - 2, '% argmax agrees % of %', [FILES[f], agree, STEPS]);
    finally
      c2.Free;
      c1.Free;
      vk.Free;
      model.Free;
      cpu.Free;
      gpu.Free;
    end;
  end;
end;

procedure TTestInferCuda.NativeClientOnDevice;
var
  dir, gguf: TFileName;
  gpu: TInferCudaBackend;
  client: TNativeLlmClient;
  llm: ILlmClient;
  msgs: TLlmMessageDynArray;
  first, req: TLlmChatRequest;
  r1, r2, r3: TLlmChatResponse;
  hits, calls: Int64;
  ctx: TQwen3Context;
begin
  if not HasDevice then
    exit;
  {$ifdef CPU32}
  AddConsole('SKIP the native client on the device: a 32-bit process');
  exit;
  {$endif CPU32}
  dir := FindQwen3Folder('model.safetensors');
  gguf := '';
  if dir <> '' then
    gguf := ExpandFileName(IncludeTrailingPathDelimiter(dir) + '..' + PathDelim +
      'qwen3-0.6b-q8_0.gguf');
  if (gguf = '') or not FileExists(gguf) then
  begin
    Check(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b-q8_0.gguf');
    AddConsole('SKIP the native client on the device: no vendor/models/' +
      'qwen3-0.6b-q8_0.gguf');
    exit;
  end;
  gpu := TInferCudaBackend.Create;
  try
    client := TNativeLlmClient.CreateFromGguf(gguf, '', gpu);
    llm := client; // the interface frees the client, and with it its Device
    client.EnableThinking := false;
    client.MaxTokens := 24;
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, 'Name three colors.');
    first := LlmChatRequest('', msgs);
    // on the backend (the model's own forward pass: TInferCudaBackend.Linear
    // per matrix), then on the device (kernels per chunk, no Linear call)
    calls := gpu.GpuLinears;
    r1 := llm.ChatComplete(first);
    Check(gpu.GpuLinears > calls, 'the backend computed the first answer');
    client.Device := TQwen3Cuda.Create(client.Model, gpu);
    Check(client.Model.Device = client.Device, 'the model computes on the device');
    ctx := client.Model.NewContext;
    try
      Check((ctx is TQwen3CuContext) and (ctx.Device = client.Device),
        'a new context is on the device');
    finally
      ctx.Free;
    end;
    calls := gpu.GpuLinears;
    r2 := llm.ChatComplete(first);
    Check(gpu.GpuLinears = calls, 'the device computed the answer, not the backend');
    Check(r2.Content <> '', 'an answer on the device');
    CheckUtf8(r2.Content = r1.Content, 'the device "%" as the backend "%"',
      [r2.Content, r1.Content]);
    // the next turn starts with the cached prompt: computed from there
    hits := client.Cache.Hits;
    SetLength(msgs, 3);
    msgs[1] := LlmMessage(lrAssistant, r2.Content);
    msgs[2] := LlmMessage(lrUser, 'One more, please.');
    req := LlmChatRequest('', msgs);
    r3 := llm.ChatComplete(req);
    Check(r3.Content <> '', 'the next turn on the device');
    Check(client.Cache.Hits = hits + 1, 'the next turn reused the device cache');
    AddConsole('native client on %: "%" / "%"', [fCu.DeviceName, r2.Content,
      r3.Content]);
    // back on the backend: the device and its contexts go
    client.Device := nil;
    Check(client.Model.Device = nil, 'no device any more');
    r1 := llm.ChatComplete(first);
    CheckUtf8(r1.Content = r2.Content, 'the backend again: "%"', [r1.Content]);
    llm := nil; // before the backend it computes with
  finally
    llm := nil;
    gpu.Free;
  end;
end;

procedure TTestInferCuda.TinyHeadDim256MatchesCpu;
const
  PROMPT = 300; // more than one chunk (256)
var
  dir: TFileName;
  gpu: TInferCudaBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  cu: TQwen3Cuda;
  c1: TQwen3Context;
  c2: TQwen3CuContext;
  ids: TIntegerDynArray;
  l1, l2: TSingleDynArray;
  i, v: PtrInt;
  seed: cardinal;
  e: double;
  id: integer;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir('tiny-qwen3-hd256');
  if CheckFailed(dir <> '', 'tiny-qwen3-hd256 found') then
    exit;
  SetLength(ids, PROMPT);
  seed := 11;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  gpu := TInferCudaBackend.Create;
  ref := TInferCpuBackend.Create;
  model := nil;
  cu := nil;
  c1 := nil;
  c2 := nil;
  try
    model := TQwen3Model.Create(dir, ref);
    v := model.Config.VocabSize;
    cu := TQwen3Cuda.Create(model, gpu);
    c1 := model.NewContext;
    c2 := cu.NewContext;
    SetLength(l1, PROMPT * v);
    SetLength(l2, PROMPT * v);
    model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
    cu.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
    e := RelError(pointer(l2), pointer(l1), PROMPT * v);
    CheckUtf8(e <= 1e-4, 'head dimension 256, prompt: relative error %', [e]);
    for i := 0 to 3 do
    begin
      id := ids[i];
      model.Forward(c1, @id, 1, pointer(l1));
      cu.Forward(c2, @id, 1, pointer(l2));
      e := RelError(pointer(l2), pointer(l1), v);
      CheckUtf8(e <= 1e-4, 'head dimension 256, decode step %: relative error %',
        [i, e]);
    end;
  finally
    c2.Free;
    c1.Free;
    cu.Free;
    model.Free;
    ref.Free;
    gpu.Free;
  end;
end;


end.
