/// tests for the Vulkan compute layer of the native inference engine (phase 4a)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - run wherever a Vulkan loader and a device exist: under WSL lavapipe (a CPU
//   implementation, mesa-vulkan-drivers), on Windows the GPU driver; elsewhere
//   they SKIP, and LAI_REQUIRE_VULKAN=1 turns that SKIP into a failure
unit test.infer.vulkan;

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
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.qwen3.vulkan,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.llm.native,
  test.infer.ref,
  test.infer.fast,
  test.infer.bpe,
  mormot.ai.infer.sampler;

type
  /// phase 4a: loader, device, buffers and a compute pipeline
  TTestInferVulkan = class(TSynTestCase)
  protected
    fVk: TInferVulkan;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) without a device
    function HasDevice: boolean;
    /// AttentionRangesMatchCpu for the tiny model in tests/data/infer-ref/aName
    procedure CheckAttentionRanges(const aName: RawUtf8);
  published
    /// y = a + b on the device: the bits of the CPU sum (an IEEE add is
    // correctly rounded on both), over a count no multiple of the work group,
    // twice with the same pipeline and other buffers
    procedure AddMatchesCpu;
    /// misuse is refused before anything reaches the driver
    procedure RefusesMisuse;
    /// device memory: what goes up comes down, at an offset, also in pieces
    procedure UploadDownload;
    /// a batch with Profile on reports its GPU time (where the queue has
    // timestamps); the tick period is read from the device limits at the
    // offset of this bitness
    procedure ProfilesBatches;
    /// phase 4b: TInferVulkanBackend.Linear for F32, Q8_0, Q4_K and Q6_K
    // weights against the reference's F32 product of the dequantized weights
    // (a tolerance: another summation order), with and without bias, more
    // than 65535 outputs, the weight cache, and the CPU for other types
    procedure LinearMatchesReference;
    /// phases 4c/4d: the tiny Qwen3 forward on the device against the CPU
    // reference - every row of a prompt longer than one chunk, decoding, a
    // reused prefix; F32 from safetensors (relative error), Q8_0 from GGUF
    // (cosine: the CPU quantizes the activations, the device does not)
    procedure TinyQwen3MatchesCpu;
    /// the tiny Qwen3 with the attention keys in one range per row (several
    // tiles of 128 keys, the online softmax rescaled) and in ranges of one
    // key (many ranges merged), against the CPU - also with a head dimension
    // of 256 (tiny-qwen3-hd256: the second accumulator of attention.comp);
    // with a KV cache of floats and of half floats (KvF16, a looser
    // tolerance), the cache grown once with tokens in it
    procedure AttentionRangesMatchCpu;
    /// Qwen3-0.6B (Q4_K_M and Q8_0 GGUF) on the device against the fast CPU
    // backend, teacher-forced: the CPU's greedy tokens fed to both, logits
    // compared at every step - skipped on a CPU implementation (lavapipe:
    // minutes), and without the model files
    procedure Qwen3GgufMatchesCpu;
    /// TNativeLlmClient on the device (Device := TQwen3Vulkan): the greedy
    // answer of Qwen3-0.6B Q8_0 as without it, the next turn from the prompt
    // cache on the device, back to the backend with Device := nil - skipped
    // as Qwen3GgufMatchesCpu
    procedure NativeClientOnDevice;
  end;


implementation

procedure TTestInferVulkan.Setup;
begin
  if TInferVulkan.Available then
  begin
    fVk := TInferVulkan.Create;
    AddConsole('Vulkan device: % (type %, vendor %)',
      [fVk.DeviceName, fVk.DeviceType, IntToHex(fVk.VendorId, 4)]);
  end;
end;

procedure TTestInferVulkan.CleanUp;
begin
  FreeAndNil(fVk);
end;

function TTestInferVulkan.HasDevice: boolean;
begin
  result := fVk <> nil;
  if result then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_VULKAN') <> '1',
    'LAI_REQUIRE_VULKAN=1 but no Vulkan device (%)', [VULKAN_LIBRARY]);
  AddConsole('SKIP Vulkan: no loader or no device (%)', [VULKAN_LIBRARY]);
end;

procedure TTestInferVulkan.AddMatchesCpu;
const
  COUNTS: array[0..1] of cardinal = (1000003, 17);
var
  pipe: TInferVkPipeline;
  a, b, y: TInferVkBuffer;
  i, k, n, bad: PtrInt;
  count: cardinal; // the push constant: a GLSL uint
  pa, pb, py: PSingleArray;
  sum: single;
begin
  if not HasDevice then
    exit;
  pipe := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  try
    for k := 0 to high(COUNTS) do
    begin
      n := COUNTS[k];
      a := nil;
      b := nil;
      y := nil;
      try
        a := TInferVkBuffer.Create(fVk, n * 4);
        b := TInferVkBuffer.Create(fVk, n * 4);
        y := TInferVkBuffer.Create(fVk, (n + 1) * 4);
        pa := a.Mapped;
        pb := b.Mapped;
        py := y.Mapped;
        for i := 0 to n - 1 do
        begin
          pa^[i] := (Random32 and $ffffff) / 4096 - 2048;
          pb^[i] := (Random32 and $ffffff) / 65536 - 128;
        end;
        // one sentinel behind the count: the shader must leave it alone
        PCardinal(@py^[n])^ := $deadbeef;
        count := n;
        fVk.Run(pipe, [a, b, y], @count, (n + 127) div 128); // local_size_x = 128
        bad := 0;
        for i := 0 to n - 1 do
        begin
          sum := pa^[i] + pb^[i];
          if PCardinal(@py^[i])^ <> PCardinal(@sum)^ then
            inc(bad);
        end;
        CheckUtf8(bad = 0, '% of % sums differ from the CPU', [bad, n]);
        Check(PCardinal(@py^[n])^ = $deadbeef, 'beyond the count');
      finally
        y.Free;
        b.Free;
        a.Free;
      end;
    end;
  finally
    pipe.Free;
  end;
end;

procedure TTestInferVulkan.RefusesMisuse;
var
  pipe: TInferVkPipeline;
  a: TInferVkBuffer;
  n: cardinal;
  raised: boolean;
begin
  if not HasDevice then
    exit;
  raised := false;
  try
    TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 6).Free;
  except
    on EInferVulkan do
      raised := true;
  end;
  Check(raised, 'push size no multiple of 4');
  pipe := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  a := nil;
  try
    a := TInferVkBuffer.Create(fVk, 16);
    n := 4;
    raised := false;
    try
      fVk.Run(pipe, [a, a], @n, 1);
    except
      on EInferVulkan do
        raised := true;
    end;
    Check(raised, 'two buffers for a pipeline of three');
    Check(fVk.MaxInvocations >= 128, 'maxComputeWorkGroupInvocations');
    Check(fVk.MaxStorageRange >= 1 shl 27, 'maxStorageBufferRange');
  finally
    a.Free;
    pipe.Free;
  end;
end;

procedure TTestInferVulkan.UploadDownload;
var
  buf: TInferVkBuffer;
  src, dst: TSingleDynArray;
  n: PtrInt;
begin
  if not HasDevice then
    exit;
  n := 300000;
  Fill(src, n, 77, 100);
  SetLength(dst, n);
  buf := TInferVkBuffer.Create(fVk, n * 4 + 64, true);
  try
    fVk.Upload(buf, 64, pointer(src), n * 4);
    fVk.Download(buf, 64, pointer(dst), n * 4);
    Check(SameBits(pointer(src), pointer(dst), n), 'round trip');
    // a part, read back from the middle
    FillCharFast(pointer(dst)^, n * 4, 0);
    fVk.Download(buf, 64 + 4000, pointer(dst), 4000);
    Check(SameBits(@src[1000], pointer(dst), 1000), 'part');
  finally
    buf.Free;
  end;
end;

procedure TTestInferVulkan.ProfilesBatches;
var
  pipe: TInferVkPipeline;
  a: TInferVkBuffer;
  n: cardinal;
  text: RawUtf8;
begin
  if not HasDevice then
    exit;
  if not fVk.HasTimestamps then
  begin
    AddConsole('SKIP Vulkan profiling: no timestamps on % ', [fVk.DeviceName]);
    exit;
  end;
  pipe := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  a := TInferVkBuffer.Create(fVk, 4 * 65536, true);
  try
    pipe.Name := 'add';
    n := 65536;
    fVk.Profile := true;
    fVk.ProfileReset;
    fVk.BatchBegin;
    fVk.BatchDispatch(pipe, [a, a, a], @n, 512);
    fVk.BatchDispatch(pipe, [a, a, a], @n, 512);
    fVk.BatchSubmit;
    text := fVk.ProfileText;
    if fVk.HasTimestamps then
      CheckUtf8(PosEx('add ', text) > 0, 'profile: %', [text])
    else
    begin
      // MoltenVK on Apple GPUs: one timestamp value for the whole batch -
      // accepted only from a portability implementation
      CheckUtf8(fVk.Portability, 'one timestamp for a whole batch on %',
        [fVk.DeviceName]);
      AddConsole('SKIP Vulkan profiling: % stamps a batch, not a dispatch',
        [fVk.DeviceName]);
    end;
  finally
    fVk.Profile := false;
    a.Free;
    pipe.Free;
  end;
end;

procedure TTestInferVulkan.LinearMatchesReference;
const
  // rows, in, out: one row, partial work, many rows, odd outputs, more
  // outputs than one dispatch dimension holds, and several gemm8.comp tiles
  // of rows and features
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 256, 1), (3, 512, 17), (37, 1024, 33), (5, 3072, 20), (1, 256, 70001),
    (70, 512, 130));
  TYPES: array[0..3] of TInferDType = (idtF32, idtQ8_0, idtQ4_K, idtQ6_K);
var
  gpu: TInferVulkanBackend;
  ref: TInferCpuBackend;
  fast: TInferCpuFastBackend;
  s, t, i, o, rows, nin, nout: PtrInt;
  kind: TInferDType;
  x, bias, wf, wd, y1, y2, y3: TSingleDynArray;
  wq: RawByteString;
  data: pointer;
  b: PSingleArray;
  e: double;
  calls: Int64;
  matvec, gemm: boolean;
begin
  if not HasDevice then
    exit;
  gpu := TInferVulkanBackend.Create;
  ref := TInferCpuBackend.Create;
  fast := TInferCpuFastBackend.Create;
  try
    // both builds of linear8.comp where the device has subgroups
    if gpu.SubgroupLinear then
      AddConsole('Vulkan subgroups: % invocations', [gpu.Vulkan.SubgroupSize])
    else
      AddConsole('Vulkan subgroups: none usable (shared-memory linear8 only)');
    // and matvec8.comp for one row of Q8_0 / K-quants, gemm8.comp or
    // linear8.comp for many rows
    matvec := gpu.UseMatvec;
    gemm := gpu.UseGemm;
    for s := ord(not gpu.SubgroupLinear) * 2 to 3 do
    begin
      gpu.UseSubgroups := s < 2;
      gpu.UseMatvec := odd(s);
      gpu.UseGemm := not odd(s);
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
    end;
    gpu.UseSubgroups := gpu.SubgroupLinear;
    gpu.UseMatvec := matvec;
    gpu.UseGemm := gemm;
    // F32 with inputs no multiple of 8: linear.comp instead of linear8.comp
    Fill(x, 3 * 36, 61, 2);
    Fill(wf, 5 * 36, 62, 0.5);
    SetLength(y1, 3 * 5);
    SetLength(y2, 3 * 5);
    ref.Linear(pointer(x), 3, 36, InferWeight(pointer(wf), idtF32), 5, nil, pointer(y1));
    calls := gpu.GpuLinears;
    gpu.Linear(pointer(x), 3, 36, InferWeight(pointer(wf), idtF32), 5, nil, pointer(y2));
    Check(gpu.GpuLinears = calls + 1, 'F32 x 36 on the device');
    e := RelError(pointer(y2), pointer(y1), 3 * 5);
    CheckUtf8(e <= 1e-5, 'F32 3 x 36 x 5: relative error %', [e]);
    // a weight type without a shader: the CPU, bit for bit as the fast backend
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

procedure TTestInferVulkan.TinyQwen3MatchesCpu;
const
  PROMPT = 300; // more than one chunk of the device forward pass (256)
var
  dir: TFileName;
  gpu: TInferVulkanBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  vk: TQwen3Vulkan;
  c1: TQwen3Context;
  c2: TQwen3VkContext;
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
    gpu := TInferVulkanBackend.Create;
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
      vk := TQwen3Vulkan.Create(model, gpu);
      vk.KvF16 := false; // the float cache: F32 within 1e-4
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
      // decoding, one token at a time
      for i := 0 to 4 do
      begin
        id := ids[i];
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

procedure TTestInferVulkan.AttentionRangesMatchCpu;
begin
  CheckAttentionRanges('tiny-qwen3');
  CheckAttentionRanges('tiny-qwen3-hd256');
end;

procedure TTestInferVulkan.CheckAttentionRanges(const aName: RawUtf8);
const
  PROMPT = 400; // two chunks; a row sees up to 400 keys: 4 tiles
  FIRST = 10;   // a first forward: the cache grows with them in it
  /// the relative error of the logits: a cache of floats, of half floats
  // (measured on an Adreno X1-85: at most 1.1e-3, the rounding of 11 bits)
  TOL: array[boolean] of double = (1e-4, 4e-3);
var
  dir: TFileName;
  gpu: TInferVulkanBackend;
  ref: TInferCpuBackend;
  model: TQwen3Model;
  vk: TQwen3Vulkan;
  c1: TQwen3Context;
  c2: TQwen3VkContext;
  ids: TIntegerDynArray;
  l1, l2, d1, d2: TSingleDynArray;
  i, k, v: PtrInt;
  seed: cardinal;
  e: double;
  id: integer;
  flash, f16: boolean;
begin
  if not HasDevice then
    exit;
  dir := FindInferRefDir(aName);
  if CheckFailed(dir <> '', Utf8ToString(aName + ' found')) then
    exit;
  SetLength(ids, PROMPT);
  seed := 11;
  for i := 0 to PROMPT - 1 do
  begin
    seed := seed * 1103515245 + 12345;
    ids[i] := 3 + (seed shr 16) mod 93;
  end;
  gpu := TInferVulkanBackend.Create;
  ref := TInferCpuBackend.Create;
  model := nil;
  vk := nil;
  c1 := nil;
  c2 := nil;
  try
    model := TQwen3Model.Create(dir, ref);
    v := model.Config.VocabSize;
    vk := TQwen3Vulkan.Create(model, gpu);
    c1 := model.NewContext;
    c2 := vk.NewContext;
    SetLength(l1, PROMPT * v);
    SetLength(l2, PROMPT * v);
    SetLength(d1, v);
    SetLength(d2, v);
    model.Forward(c1, pointer(ids), PROMPT, pointer(l1), true);
    flash := vk.UseFlash;
    for f16 := false to true do
    begin
      vk.KvF16 := f16;
      // flash.comp (where the device runs it) and attention.comp for the
      // prompt; a Reset context takes the cache format of KvF16
      for k := 0 to 3 do
      begin
        vk.SplitKeys := k and 1; // 0: one range per row, 1: a range per key
        vk.UseFlash := vk.HasFlash and (k < 2);
        c2.Reset;
        if k = 0 then
        begin
          // a new cache (64 tokens: c2 is new, or its cache dropped for the
          // other format), grown to the prompt with FIRST in it
          vk.Forward(c2, pointer(ids), FIRST, pointer(l2), true);
          vk.Forward(c2, @ids[FIRST], PROMPT - FIRST, @l2[FIRST * v], true);
        end
        else
          vk.Forward(c2, pointer(ids), PROMPT, pointer(l2), true);
        Check(c2.KvF16 = f16, 'the cache format of KvF16');
        e := RelError(pointer(l2), pointer(l1), PROMPT * v);
        CheckUtf8(e <= TOL[f16],
          '% KvF16=% SplitKeys=% UseFlash=% prompt: relative error %',
          [aName, f16, k and 1, vk.UseFlash, e]);
      end;
      vk.UseFlash := flash;
      // decoding past the prompt, one range and many ranges in turn
      for i := 0 to 3 do
      begin
        vk.SplitKeys := i and 1;
        id := ids[i];
        model.Forward(c1, @id, 1, pointer(d1));
        vk.Forward(c2, @id, 1, pointer(d2));
        e := RelError(pointer(d2), pointer(d1), v);
        CheckUtf8(e <= TOL[f16],
          '% KvF16=% SplitKeys=% decode step %: relative error %',
          [aName, f16, i and 1, i, e]);
      end;
      c1.Truncate(PROMPT);
    end;
  finally
    c2.Free;
    c1.Free;
    vk.Free;
    model.Free;
    ref.Free;
    gpu.Free;
  end;
end;

procedure TTestInferVulkan.Qwen3GgufMatchesCpu;
const
  FILES: array[0..1] of TFileName = ('qwen3-0.6b-q4_k_m.gguf', 'qwen3-0.6b-q8_0.gguf');
  // "Wie verlege ich ein Kupferrohr?" - any ids do; these are ordinary words
  PROMPT: array[0..11] of integer = (
    151644, 872, 198, 54, 645, 2739, 273, 10109, 12153, 3105, 63084, 30);
  STEPS = 16;
var
  dir, gguf: TFileName;
  f, i, v, agree: PtrInt;
  gpu: TInferVulkanBackend;
  cpu: TInferCpuFastBackend;
  model: TQwen3Model;
  vk: TQwen3Vulkan;
  c1: TQwen3Context;
  c2: TQwen3VkContext;
  l1, l2: TSingleDynArray;
  cmp: TInferCompare;
  worst: double;
  id: integer;
begin
  if not HasDevice then
    exit;
  if fVk.DeviceType = VK_PHYSICAL_DEVICE_TYPE_CPU then
  begin
    AddConsole('SKIP Qwen3-0.6B on the device: % is a CPU implementation',
      [fVk.DeviceName]);
    exit;
  end;
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
    gpu := TInferVulkanBackend.Create;
    cpu := TInferCpuFastBackend.Create;
    model := nil;
    vk := nil;
    c1 := nil;
    c2 := nil;
    try
      model := TQwen3Model.CreateFromGguf(gguf, cpu);
      v := model.Config.VocabSize;
      vk := TQwen3Vulkan.Create(model, gpu);
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
        [FILES[f], fVk.DeviceName, worst, agree, STEPS]);
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

procedure TTestInferVulkan.NativeClientOnDevice;
var
  dir, gguf: TFileName;
  gpu: TInferVulkanBackend;
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
  if fVk.DeviceType = VK_PHYSICAL_DEVICE_TYPE_CPU then
  begin
    AddConsole('SKIP the native client on the device: % is a CPU implementation',
      [fVk.DeviceName]);
    exit;
  end;
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
  gpu := TInferVulkanBackend.Create;
  try
    client := TNativeLlmClient.CreateFromGguf(gguf, '', gpu);
    llm := client; // the interface frees the client, and with it its Device
    client.EnableThinking := false;
    client.MaxTokens := 24;
    SetLength(msgs, 1);
    msgs[0] := LlmMessage(lrUser, 'Name three colors.');
    first := LlmChatRequest('', msgs);
    // on the backend (the model's own forward pass: TInferVulkanBackend.Linear
    // per matrix), then on the device (one batch per chunk, no Linear call)
    calls := gpu.GpuLinears;
    r1 := llm.ChatComplete(first);
    Check(gpu.GpuLinears > calls, 'the backend computed the first answer');
    client.Device := TQwen3Vulkan.Create(client.Model, gpu);
    // the float cache: the same words as the backend over a whole answer (the
    // half floats of KvF16 may flip a close greedy choice; Qwen3GgufMatchesCpu
    // checks them)
    TQwen3Vulkan(client.Device).KvF16 := false;
    Check(client.Model.Device = client.Device, 'the model computes on the device');
    ctx := client.Model.NewContext;
    try
      Check((ctx is TQwen3VkContext) and (ctx.Device = client.Device),
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
    AddConsole('native client on %: "%" / "%"', [fVk.DeviceName, r2.Content,
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

end.
