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
  mormot.ai.infer.cuda;

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
  end;


implementation

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
  buf: TInferCuBuffer;
  up, down: array[0..N - 1] of byte;
  i: PtrInt;
begin
  if not HasDevice then
    exit;
  for i := 0 to N - 1 do
    up[i] := byte(i * 7 + 3);
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
  finally
    buf.Free;
  end;
end;


end.
