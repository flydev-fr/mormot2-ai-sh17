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
  mormot.ai.infer.vulkan;

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

end.
