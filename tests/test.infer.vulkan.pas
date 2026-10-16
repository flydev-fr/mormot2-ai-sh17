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
  test.infer.fast;

type
  /// phase 4a: loader, device, buffers and a compute pipeline
  TTestInferVulkan = class(TSynTestCase)
  protected
    fVk: TInferVulkan;
    procedure Setup; override;
    procedure CleanUp; override;
    /// false (and SKIP or a failure) without a device
    function HasDevice: boolean;
  published
    /// y = a + b on the device: the bits of the CPU sum (an IEEE add is
    // correctly rounded on both), over a count no multiple of the work group,
    // twice with the same pipeline and other buffers
    procedure AddMatchesCpu;
    /// misuse is refused before anything reaches the driver
    procedure RefusesMisuse;
    /// device memory: what goes up comes down, at an offset, also in pieces
    procedure UploadDownload;
    /// phase 4b: TInferVulkanBackend.Linear for F32, Q8_0, Q4_K and Q6_K
    // weights against the reference's F32 product of the dequantized weights
    // (a tolerance: another summation order), with and without bias, more
    // than 65535 outputs, the weight cache, and the CPU for other types
    procedure LinearMatchesReference;
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

procedure TTestInferVulkan.LinearMatchesReference;
const
  // rows, in, out: one row, partial work, many rows, odd outputs, and more
  // outputs than one dispatch dimension holds
  SHAPES: array[0..4, 0..2] of integer = (
    (1, 256, 1), (3, 512, 17), (37, 1024, 33), (5, 3072, 20), (1, 256, 70001));
  TYPES: array[0..3] of TInferDType = (idtF32, idtQ8_0, idtQ4_K, idtQ6_K);
var
  gpu: TInferVulkanBackend;
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
  gpu := TInferVulkanBackend.Create;
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

end.
