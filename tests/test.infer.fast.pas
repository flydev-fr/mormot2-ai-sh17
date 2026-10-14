/// tests for the fast CPU backend of the native inference engine (phase 1c)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - two contracts of TInferCpuFastBackend against the scalar reference:
//   * Exact = true: the very bits of TInferCpuBackend - work split by output
//     feature, head or element range, never inside one sum; the SIMD kernels
//     return exact integer block sums and scale them in the reference's order
//   * Exact = false (default): float arithmetic where the CPU has a kernel (Q8_0
//     Linear in float lanes like ggml, float attention, a GELU table), checked
//     against the reference within stated tolerances; the activation quantizer
//     and the integer block sums stay exact in both modes
// - one arithmetic per backend whatever the load: a caller that finds the pool
//   busy computes on its own thread with the same kernels, bit for bit
// - the thread pool: every item runs exactly once, also under a burst of tiny
//   jobs, an exception reaches the caller, two callers at once both get the
//   right result
unit test.infer.fast;

interface

{$I mormot.defines.inc}

uses
  classes,
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.threads,
  mormot.ai.infer.model.xlmr,
  mormot.ai.infer.pipeline.embed,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.embed.native,
  test.infer.ref,
  test.infer.gguf;

type
  /// phase 1c: the fast backend against the reference, bit for bit
  TTestInferFast = class(TSynTestCase)
  protected
    fRef: TInferCpuBackend;
    fFast, fExact: TInferCpuFastBackend;
    fHits: TIntegerDynArray;
    fFail: integer;
    procedure Setup; override;
    procedure CleanUp; override;
    procedure CountHit(aItem: integer);
    procedure FailOne(aItem: integer);
  published
    /// every kernel of this CPU against the plain Pascal loop
    procedure KernelsMatchPascal;
    /// exact mode: Linear F32 and Q8_0, odd shapes included: the same bytes
    procedure LinearIsBitIdentical;
    /// exact mode: Attention (with a mask), GELU and LayerNorm (both modes):
    // the same bytes as the reference
    procedure ElementwiseOpsAreBitIdentical;
    /// default mode: Linear, Attention and GELU within their tolerances
    procedure FastModeWithinTolerance;
    /// the thread pool: exactly once, bursts, exceptions
    procedure ThreadPoolRunsEveryItemOnce;
    /// two threads embedding at once through one fast backend
    procedure ConcurrentCallersAreCorrect;
    /// tiny GGUF and, if present, BGE-M3 Q8: exact mode equal to the reference,
    // default mode within tolerance
    procedure ForwardIsBitIdentical;
    /// the IEmbedder adapter: shared per name, Embed = EmbedBatch, long text refused
    procedure NativeEmbedderAdapter;
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
    aValues[i] := ((aSeed shr 8) / 16777216 * 2 - 1) * aScale;
  end;
end;

/// Q8_0 weights [aOut, aIn] with random scales and every int8 value
function RandomQ8(aOut, aIn: PtrInt; aSeed: cardinal): RawByteString;
var
  b, j: PtrInt;
  blk: PInferBlockQ8_0;
begin
  SetLength(result, aOut * (aIn div 32) * SizeOf(TInferBlockQ8_0));
  blk := pointer(result);
  for b := 0 to aOut * (aIn div 32) - 1 do
  begin
    aSeed := aSeed * 1103515245 + 12345;
    blk^.d := InferSingleToHalf(((aSeed shr 8) and 1023) / 65536 + 1e-4);
    for j := 0 to 31 do
    begin
      aSeed := aSeed * 1103515245 + 12345;
      blk^.qs[j] := shortint(aSeed shr 24); // -128 .. 127
    end;
    inc(blk);
  end;
end;

function SameBits(a, b: PSingleArray; aCount: PtrInt): boolean;
begin
  result := CompareMem(a, b, aCount * SizeOf(single));
end;


{ TTestInferFast }

procedure TTestInferFast.Setup;
begin
  fRef := TInferCpuBackend.Create;
  fFast := TInferCpuFastBackend.Create(4); // a pool even on a 1-core runner
  fExact := TInferCpuFastBackend.Create(4);
  fExact.Exact := true;
end;

procedure TTestInferFast.CleanUp;
begin
  FreeAndNil(fExact); // re-entrant, as mORMot requires
  FreeAndNil(fFast);
  FreeAndNil(fRef);
end;

function DotD(a, b: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
  v: double;
begin
  result := 0;
  for i := 0 to n - 1 do
  begin
    v := a^[i];
    result := result + v * b^[i];
  end;
end;

function SumAbs(a, b: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
  v: double;
begin
  result := 0;
  for i := 0 to n - 1 do
  begin
    v := a^[i];
    result := result + Abs(v * b^[i]);
  end;
end;

/// sum over j of p[j] * v[j * n + d]
function AxpyD(p, v: PSingleArray; n, aKeys, d: PtrInt): double;
var
  j: PtrInt;
  a: double;
begin
  result := 0;
  for j := 0 to aKeys - 1 do
  begin
    a := p^[j];
    result := result + a * v^[j * n + d];
  end;
end;

/// max |a - b| relative to max |b| - a scale-free error of a whole output
function RelError(a, b: PSingleArray; aCount: PtrInt): double;
var
  i: PtrInt;
  m, e: double;
begin
  m := 0;
  e := 0;
  for i := 0 to aCount - 1 do
  begin
    // a NaN or an infinity on either side is an error, not a comparison that
    // happens to be false
    if (PCardinal(@a^[i])^ and $7f800000 = $7f800000) or
       (PCardinal(@b^[i])^ and $7f800000 = $7f800000) then
    begin
      result := 1e30;
      exit;
    end;
    if Abs(b^[i]) > m then
      m := Abs(b^[i]);
    if Abs(a^[i] - b^[i]) > e then
      e := Abs(a^[i] - b^[i]);
  end;
  if m = 0 then
    result := e
  else
    result := e / m;
end;

procedure TTestInferFast.KernelsMatchPascal;
var
  xs, d1, d2, g: TSingleDynArray;
  q1, q2: RawByteString;
  r, j, n: PtrInt;
  w, x: RawByteString;
  nb, b, k: PtrInt;
  s1, s2: TIntegerDynArray;
  dx, acc, accRef, magn: TDoubleDynArray;
  dxs, out8: TSingleDynArray;
  wb: PInferBlockQ8_0;
  a8: double;
begin
  AddConsole(Utf8ToString('Q8_0 kernel: ' + InferQ8KernelName));
  // the activation quantizer of this CPU against the Pascal loop: same bytes,
  // ties, a zero block, subnormal scales, large values included
  for r := 0 to 8 do
  begin
    Fill(xs, 32 * 9, 900 + r, 10);
    case r of
      1:
        for j := 0 to 31 do
          xs[j] := 0;
      2:
        for j := 32 to 63 do
          xs[j] := xs[j] * 1e-38;
      3:
        for j := 0 to high(xs) do
          xs[j] := xs[j] * 1e30;
      4:
        begin
          xs[0] := 127;
          for j := 1 to 31 do
            xs[j] := j - 15.5; // halves: ties to even
        end;
      5:
        begin
          for j := 64 to 95 do
            xs[j] := xs[j] * 1e-41;
          // a scale just above 2^-128: 1 / d close to the largest single
          for j := 96 to 127 do
            xs[j] := xs[j] * 5e-37;
        end;
      6:
        // a valid scale (787) from values beyond the largest half: no
        // intermediate of the quantizer may overflow on the way
        for j := 0 to 63 do
          xs[j] := 100000 - j * 1000;
      7, 8:
        begin
          // the zero threshold itself: amax = 127 * 2^-128 gives d = 2^-128
          // exactly (all q = 0); four ulps more give d = 2^-128 + 2^-149, the
          // first scale above it (one ulp still rounds back to 2^-128)
          for j := 32 to 63 do
            xs[j] := xs[j] * 1e-40;
          PCardinal(@xs[32])^ := $02FE0000 + 4 * cardinal(r - 7);
        end;
    end;
    SetLength(q1, 9 * SizeOf(TInferBlockQ8_0));
    SetLength(q2, 9 * SizeOf(TInferBlockQ8_0));
    InferQuantizeRowQ8_0(pointer(xs), length(xs), pointer(q1));
    InferQuantizeRowQ8_0Pascal(pointer(xs), length(xs), pointer(q2));
    CheckUtf8(q1 = q2, 'quantizer = Pascal, row %', [r]);
    if r >= 7 then
      // block 1 starts with its largest value: q = 0 at the threshold, 127 above
      CheckEqual(PInferBlockQ8_0Array(pointer(q1))^[1].qs[0], 127 * (r - 7),
        'quantizer at the zero threshold');
  end;
  if Assigned(InferQ8QuantizeRow) then
  begin
    // a NaN in block 3 of 9: status 1 + 3, blocks 0..2 written as by Pascal,
    // nothing written for block 3 or after it
    Fill(xs, 32 * 9, 950, 10);
    InferQuantizeRowQ8_0Pascal(pointer(xs), 3 * 32, pointer(q2));
    PCardinal(@xs[3 * 32 + 17])^ := $7fc00000;
    q1 := '';
    SetLength(q1, 9 * SizeOf(TInferBlockQ8_0));
    FillCharFast(pointer(q1)^, length(q1), $AA);
    j := InferQ8QuantizeRow(pointer(xs), 9, pointer(q1));
    CheckEqual(j, 4, 'quantizer status of a NaN in block 3');
    CheckUtf8(CompareMem(pointer(q1), pointer(q2), 3 * SizeOf(TInferBlockQ8_0)),
      'quantizer: the blocks before the bad one', []);
    for j := 3 * SizeOf(TInferBlockQ8_0) to length(q1) - 1 do
      if q1[j + 1] <> #$AA then
      begin
        CheckUtf8(false, 'quantizer wrote byte % after the bad block', [j]);
        break;
      end;
  end;
  // the float attention kernels against double sums
  if Assigned(InferDotsF32) then
  begin
    // rows of n values, n + 16 apart (as heads inside a wider row)
    n := 64;
    Fill(xs, (n + 16) * 7, 31, 2);
    Fill(d1, n, 32, 2);
    SetLength(d2, 8);
    d2[7] := 12345;
    InferDotsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 7, n,
      pointer(d2));
    for j := 0 to 6 do
      CheckUtf8(Abs(d2[j] - DotD(pointer(d1), @xs[j * (n + 16)], n)) <=
        1e-5 * SumAbs(pointer(d1), @xs[j * (n + 16)], n), 'float dot %', [j]);
    CheckUtf8(d2[7] = 12345, 'float dot: nothing beyond the keys', []);
    d2[0] := 12345;
    InferDotsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 0, n,
      pointer(d2));
    CheckUtf8(d2[0] = 12345, 'float dot: no key, nothing written', []);
    SetLength(d2, n + 1);
    FillCharFast(pointer(d2)^, n * SizeOf(single), 0);
    d2[n] := 12345;
    InferAxpyRowsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 7, n,
      pointer(d2));
    for j := 0 to n - 1 do
      CheckUtf8(Abs(d2[j] - AxpyD(pointer(d1), pointer(xs), n + 16, 7, j)) <= 1e-4,
        'float axpy %', [j]);
    CheckUtf8(d2[n] = 12345, 'float axpy: nothing beyond the row', []);
    g := copy(d2);
    InferAxpyRowsF32(pointer(d1), pointer(xs), (n + 16) * SizeOf(single), 0, n,
      pointer(d2));
    CheckUtf8(SameBits(pointer(g), pointer(d2), n + 1),
      'float axpy: no key, nothing written', []);
  end;
  for nb := 0 to 9 do
  begin
    w := RandomQ8(4, nb * 32, 7 + nb);
    x := RandomQ8(1, nb * 32, 99 + nb);
    SetLength(s1, nb * 4 + 1);
    SetLength(s2, nb * 4 + 1);
    InferQ8BlockDots(pointer(w), pointer(x), nb, pointer(s1));
    InferQ8BlockDotsPascal(pointer(w), pointer(x), nb, pointer(s2));
    CheckUtf8(CompareMem(pointer(s1), pointer(s2), nb * 4), '1-row kernel, % blocks', [nb]);
    InferQ8BlockDots4(pointer(w), pointer(x), nb, pointer(s1),
      nb * SizeOf(TInferBlockQ8_0));
    InferQ8BlockDots4Pascal(pointer(w), pointer(x), nb, pointer(s2),
      nb * SizeOf(TInferBlockQ8_0));
    CheckUtf8(CompareMem(pointer(s1), pointer(s2), nb * 16), '4-row kernel, % blocks', [nb]);
    SetLength(dx, nb + 1);
    for b := 0 to nb - 1 do
      dx[b] := InferHalfToSingle(PInferBlockQ8_0Array(pointer(x))^[b].d);
    if Assigned(InferQ8Linear4) then
    begin
      // the fused kernel against the reference arithmetic, row by row
      SetLength(acc, 4);
      SetLength(accRef, 4);
      for k := 0 to 3 do
      begin
        acc[k] := k - 1.5; // a bias
        accRef[k] := acc[k];
        for b := 0 to nb - 1 do
        begin
          wb := @PInferBlockQ8_0Array(pointer(w))^[k * nb + b];
          accRef[k] := accRef[k] + s2[b * 4 + k] * (InferHalfToSingle(wb^.d) * dx[b]);
        end;
      end;
      InferQ8Linear4(pointer(w), pointer(x), nb, pointer(dx),
        nb * SizeOf(TInferBlockQ8_0), pointer(acc));
      CheckUtf8(CompareMem(pointer(acc), pointer(accRef), 4 * SizeOf(double)),
        'fused kernel = reference arithmetic, % blocks', [nb]);
    end;
    if not Assigned(InferQ8DotF32x4x2) then
      continue;
    // the float kernel: two different activation rows, within float rounding
    // of the exact double sums (no bias); activations as the quantizer writes
    // them, -127 .. 127 (the AVX2 kernel relies on it)
    x := RandomQ8(2, nb * 32, 199 + nb);
    for b := 0 to nb * 2 - 1 do
      for j := 0 to 31 do
        if PInferBlockQ8_0Array(pointer(x))^[b].qs[j] = -128 then
          PInferBlockQ8_0Array(pointer(x))^[b].qs[j] := -127;
    SetLength(dxs, nb * 2 + 1);
    for b := 0 to nb * 2 - 1 do
      dxs[b] := InferHalfToSingle(PInferBlockQ8_0Array(pointer(x))^[b].d);
    SetLength(out8, 8);
    InferQ8DotF32x4x2(pointer(w), pointer(x),
      @PInferBlockQ8_0Array(pointer(x))^[nb], nb, pointer(dxs), @dxs[nb],
      nb * SizeOf(TInferBlockQ8_0), pointer(out8));
    SetLength(accRef, 8);
    SetLength(magn, 8);
    for r := 0 to 1 do
    begin
      InferQ8BlockDots4Pascal(pointer(w), @PInferBlockQ8_0Array(pointer(x))^[r * nb],
        nb, pointer(s2), nb * SizeOf(TInferBlockQ8_0));
      for k := 0 to 3 do
      begin
        accRef[r * 4 + k] := 0;
        magn[r * 4 + k] := 0;
        for b := 0 to nb - 1 do
        begin
          wb := @PInferBlockQ8_0Array(pointer(w))^[k * nb + b];
          a8 := s2[b * 4 + k] * (InferHalfToSingle(wb^.d) * dxs[r * nb + b]);
          accRef[r * 4 + k] := accRef[r * 4 + k] + a8;
          magn[r * 4 + k] := magn[r * 4 + k] + Abs(a8);
        end;
      end;
    end;
    for k := 0 to 7 do
      CheckUtf8(Abs(out8[k] - accRef[k]) <= 1e-5 * (magn[k] + 1),
        'float kernel output %, % blocks: % <> %', [k, nb, out8[k], accRef[k]]);
  end;
end;

procedure TTestInferFast.LinearIsBitIdentical;
const
  SHAPES: array[0..5, 0..2] of integer = (
    (1, 32, 1), (3, 64, 5), (7, 96, 17), (37, 256, 33), (64, 1024, 100),
    (5, 4096, 64));
var
  i: PtrInt;
  x, wf, bias, y1, y2: TSingleDynArray;
  wq: RawByteString;
  rows, nin, nout: integer;
begin
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 11 + i, 3);
    Fill(wf, nout * nin, 23 + i, 0.5);
    Fill(bias, nout, 31 + i, 1);
    wq := RandomQ8(nout, nin, 41 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wf), idtF32), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wf), idtF32), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'F32 % x % x %',
      [rows, nin, nout]);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q8_0 % x % x %',
      [rows, nin, nout]);
    // without bias, too
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      nil, pointer(y1));
    fExact.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      nil, pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), rows * nout), 'Q8_0 no bias % x % x %',
      [rows, nin, nout]);
  end;
end;

procedure TTestInferFast.ElementwiseOpsAreBitIdentical;
var
  q, k, v, y1, y2, s, g1, g2: TSingleDynArray;
  keep: TIntegerDynArray;
  seq, heads, hd, i: PtrInt;
begin
  for seq := 1 to 40 do
    if seq in [1, 2, 15, 16, 17, 40] then
    begin
      heads := 4;
      hd := 8;
      Fill(q, seq * heads * hd, seq, 2);
      Fill(k, seq * heads * hd, seq + 100, 2);
      Fill(v, seq * heads * hd, seq + 200, 2);
      SetLength(keep, seq);
      for i := 0 to seq - 1 do
        keep[i] := ord((i = 0) or (i mod 5 <> 3)); // a few masked keys
      SetLength(s, seq * seq);
      SetLength(y1, seq * heads * hd);
      SetLength(y2, seq * heads * hd);
      fRef.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y1));
      fExact.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y2));
      CheckUtf8(SameBits(pointer(y1), pointer(y2), seq * heads * hd),
        'attention, % tokens', [seq]);
    end;
  for i := 0 to 2 do
  begin
    Fill(g1, 5000 + i * 7000, 77 + i, 12); // beyond and below the split size
    g2 := copy(g1);
    fRef.Gelu(pointer(g1), length(g1));
    fExact.Gelu(pointer(g2), length(g2));
    CheckUtf8(SameBits(pointer(g1), pointer(g2), length(g1)), 'gelu of %',
      [length(g1)]);
  end;
  // LayerNorm: rows split over the pool, every row in the reference
  // arithmetic - in both modes; below, at and above the parallel threshold,
  // into another buffer and in place
  for i := 0 to 4 do
  begin
    seq := 1 + i * 16; // 1, 17, 33, 49, 65 rows
    Fill(g1, seq * 1024, 300 + i, 4);
    Fill(q, 1024, 400 + i, 1);
    Fill(k, 1024, 500 + i, 1);
    SetLength(y1, length(g1));
    SetLength(y2, length(g1));
    fRef.LayerNorm(pointer(g1), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(y1));
    fFast.LayerNorm(pointer(g1), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(y2));
    CheckUtf8(SameBits(pointer(y1), pointer(y2), length(y1)),
      'layernorm, % rows', [seq]);
    g2 := copy(g1);
    fExact.LayerNorm(pointer(g2), seq, 1024, pointer(q), pointer(k), 1e-5,
      pointer(g2));
    CheckUtf8(SameBits(pointer(y1), pointer(g2), length(y1)),
      'layernorm in place, % rows', [seq]);
  end;
end;

procedure TTestInferFast.FastModeWithinTolerance;
const
  SHAPES: array[0..4, 0..2] of integer = (
    (1, 32, 4), (3, 64, 5), (37, 256, 33), (64, 1024, 100), (5, 4096, 64));
var
  i, j, rows, nin, nout, seq, heads, hd: PtrInt;
  x, bias, y1, y2, q, k, v, s, g1, g2: TSingleDynArray;
  keep: TIntegerDynArray;
  wq: RawByteString;
  e, worst: double;
  refused: boolean;
begin
  // Q8_0 Linear: float lanes against the double reference - a few float ulps
  // of the largest output
  worst := 0;
  for i := 0 to high(SHAPES) do
  begin
    rows := SHAPES[i, 0];
    nin := SHAPES[i, 1];
    nout := SHAPES[i, 2];
    Fill(x, rows * nin, 111 + i, 3);
    Fill(bias, nout, 131 + i, 1);
    wq := RandomQ8(nout, nin, 141 + i);
    SetLength(y1, rows * nout);
    SetLength(y2, rows * nout);
    fRef.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y1));
    fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
      pointer(bias), pointer(y2));
    e := RelError(pointer(y2), pointer(y1), rows * nout);
    if e > worst then
      worst := e;
    CheckUtf8(e <= 2e-6, 'Q8_0 % x % x %: relative error %', [rows, nin, nout, e]);
  end;
  AddConsole(FormatString('fast Q8_0 Linear: worst relative error %', [worst]));
  // attention: float scores and sums
  heads := 4;
  hd := 64;
  for seq := 1 to 40 do
    if seq in [1, 3, 17, 40] then
    begin
      Fill(q, seq * heads * hd, seq + 300, 1);
      Fill(k, seq * heads * hd, seq + 400, 1);
      Fill(v, seq * heads * hd, seq + 500, 2);
      SetLength(keep, seq);
      for j := 0 to seq - 1 do
        keep[j] := ord((j = 0) or (j mod 5 <> 3));
      SetLength(s, seq * seq);
      SetLength(y1, seq * heads * hd);
      SetLength(y2, seq * heads * hd);
      fRef.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y1));
      fFast.Attention(pointer(q), pointer(k), pointer(v), seq, heads, hd,
        pointer(keep), pointer(s), pointer(y2));
      e := RelError(pointer(y2), pointer(y1), seq * heads * hd);
      CheckUtf8(e <= 1e-5, 'attention % tokens: relative error %', [seq, e]);
    end;
  // GELU table: absolute error far below the single precision of the result
  Fill(g1, 20000, 77, 12);
  g2 := copy(g1);
  fRef.Gelu(pointer(g1), length(g1));
  fFast.Gelu(pointer(g2), length(g2));
  // table error (< 1e-9) plus at most one single rounding step of the result
  worst := 0;
  for j := 0 to high(g1) do
  begin
    // a NaN or an infinity is a failure, not a comparison that happens to be false
    if (PCardinal(@g1[j])^ and $7f800000 = $7f800000) or
       (PCardinal(@g2[j])^ and $7f800000 = $7f800000) then
    begin
      worst := 1e30;
      break;
    end;
    e := Abs(g2[j] - g1[j]) - 1.2e-7 * Abs(g1[j]);
    if e > worst then
      worst := e;
  end;
  CheckUtf8(worst <= 1e-9, 'gelu table: error % beyond one single ulp', [worst]);
  // an activation block whose FP16 scale is +Inf (|x| >= 127 * 65520) is
  // refused before any kernel runs - on this thread and on the pool
  for i := 0 to 1 do
  begin
    rows := 1 + i * 63;
    nin := 1024;
    nout := 100;
    Fill(x, rows * nin, 171 + i, 3);
    x[(rows - 1) * nin + 40] := 1e7;
    wq := RandomQ8(nout, nin, 181 + i);
    SetLength(y2, rows * nout);
    refused := false;
    try
      fFast.Linear(pointer(x), rows, nin, InferWeight(pointer(wq), idtQ8_0), nout,
        nil, pointer(y2));
    except
      on E: EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'Linear refuses an activation scale of +Inf, % rows', [rows]);
  end;
end;

procedure TTestInferFast.CountHit(aItem: integer);
begin
  InterlockedIncrement(fHits[aItem]);
end;

procedure TTestInferFast.FailOne(aItem: integer);
begin
  if aItem = 13 then
    EInferError.RaiseUtf8('item %', [aItem]);
  InterlockedIncrement(fHits[aItem]);
end;

procedure TTestInferFast.ThreadPoolRunsEveryItemOnce;
var
  pool: TInferThreadPool;
  run, n, i, bad: integer;
  raised: boolean;
begin
  pool := TInferThreadPool.Create(5);
  try
    bad := 0;
    // a burst of jobs of every size: tickets of one job never leak into the next
    for run := 1 to 3000 do
    begin
      n := 1 + (run * 7919) mod 97;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once over 3000 jobs');
    // sizes 8, 2, 8, 1 in turn: a late worker holding an older snapshot must
    // never mix one job's Count with another's Base (seqlock, not Count twice)
    bad := 0;
    for run := 1 to 20000 do
    begin
      case run and 3 of
        0, 2:
          n := 8;
        1:
          n := 2;
      else
        n := 1;
      end;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once over 20000 jobs of 8, 2, 8, 1');
    // the ticket counter restarts before it wraps: jobs across the restart
    pool.Ticket := $40000000 - 100;
    bad := 0;
    for run := 1 to 200 do
    begin
      n := 1 + run mod 9;
      SetLength(fHits, n);
      FillCharFast(pointer(fHits)^, n * SizeOf(integer), 0);
      pool.Run(CountHit, n);
      for i := 0 to n - 1 do
        if fHits[i] <> 1 then
          inc(bad);
    end;
    CheckEqual(bad, 0, 'every item exactly once across the ticket restart');
    CheckUtf8(cardinal(pool.Ticket) < $40000000, 'ticket counter restarted');
    // an exception in one item reaches the caller; the others still ran
    SetLength(fHits, 50);
    FillCharFast(pointer(fHits)^, 50 * SizeOf(integer), 0);
    raised := false;
    try
      pool.Run(FailOne, 50);
    except
      on E: EInferError do
        raised := PosEx('item 13', StringToUtf8(E.Message)) > 0;
    end;
    CheckUtf8(raised, 'the item exception reaches Run');
    bad := 0;
    for i := 0 to 49 do
      if (i <> 13) and (fHits[i] <> 1) then
        inc(bad);
    CheckEqual(bad, 0, 'the other items ran once');
    // the pool is usable after an exception
    SetLength(fHits, 20);
    FillCharFast(pointer(fHits)^, 20 * SizeOf(integer), 0);
    pool.Run(CountHit, 20);
    bad := 0;
    for i := 0 to 19 do
      if fHits[i] <> 1 then
        inc(bad);
    CheckEqual(bad, 0, 'next job after an exception');
  finally
    pool.Free;
  end;
end;

type
  /// computes one Q8_0 Linear in a loop through a shared backend
  TLinearCaller = class(TThread)
  protected
    fBackend: TInferBackend;
    fX, fBias: TSingleDynArray;
    fW: RawByteString;
    fRows, fIn, fOut: integer;
    procedure Execute; override;
  public
    Result: TSingleDynArray;
    Mismatch: boolean;
    Done: integer;
    Expected: TSingleDynArray;
  end;

procedure TLinearCaller.Execute;
var
  i: integer;
begin
  SetLength(Result, fRows * fOut);
  for i := 1 to 30 do
  begin
    fBackend.Linear(pointer(fX), fRows, fIn, InferWeight(pointer(fW), idtQ8_0),
      fOut, pointer(fBias), pointer(Result));
    if not SameBits(pointer(Result), pointer(Expected), fRows * fOut) then
      Mismatch := true;
    inc(Done);
  end;
end;

type
  /// computes one Attention in a loop through a shared backend
  TAttentionCaller = class(TThread)
  protected
    fBackend: TInferBackend;
    fQ, fK, fV, fS: TSingleDynArray;
    fKeep: TIntegerDynArray;
    fSeq, fHeads, fDim: integer;
    procedure Execute; override;
  public
    Result: TSingleDynArray;
    Mismatch: boolean;
    Done: integer;
    Expected: TSingleDynArray;
  end;

procedure TAttentionCaller.Execute;
var
  i: integer;
begin
  SetLength(Result, fSeq * fHeads * fDim);
  SetLength(fS, fSeq * fSeq);
  for i := 1 to 40 do
  begin
    fBackend.Attention(pointer(fQ), pointer(fK), pointer(fV), fSeq, fHeads, fDim,
      pointer(fKeep), pointer(fS), pointer(Result));
    if not SameBits(pointer(Result), pointer(Expected), length(Result)) then
      Mismatch := true;
    inc(Done);
  end;
end;

procedure TTestInferFast.ConcurrentCallersAreCorrect;
var
  c: array[0..1] of TLinearCaller;
  a: array[0..1] of TAttentionCaller;
  i, j: PtrInt;
  sc: TSingleDynArray;
  alone: TInferCpuFastBackend;
begin
  // the expected bits: the fast arithmetic on one thread - a caller finding the
  // pool busy must compute exactly that, not fall back to another arithmetic
  alone := TInferCpuFastBackend.Create(1);
  try
  for i := 0 to 1 do
  begin
    c[i] := TLinearCaller.Create(true);
    c[i].fBackend := fFast;
    c[i].fRows := 9 + i;
    c[i].fIn := 256;
    c[i].fOut := 96 + i * 32;
    Fill(c[i].fX, c[i].fRows * c[i].fIn, 500 + i, 2);
    Fill(c[i].fBias, c[i].fOut, 600 + i, 1);
    c[i].fW := RandomQ8(c[i].fOut, c[i].fIn, 700 + i);
    SetLength(c[i].Expected, c[i].fRows * c[i].fOut);
    alone.Linear(pointer(c[i].fX), c[i].fRows, c[i].fIn,
      InferWeight(pointer(c[i].fW), idtQ8_0), c[i].fOut, pointer(c[i].fBias),
      pointer(c[i].Expected));
  end;
  finally
    alone.Free;
  end;
  try
    for i := 0 to 1 do
      {$ifdef HASTTHREADSTART} c[i].Start; {$else} c[i].Resume; {$endif}
    for i := 0 to 1 do
      c[i].WaitFor;
    for i := 0 to 1 do
      CheckUtf8(not c[i].Mismatch and (c[i].Done = 30) and (c[i].FatalException = nil),
        'caller % always right (pool or own thread), every round done', [i]);
  finally
    for i := 0 to 1 do
      c[i].Free;
  end;
  // exact mode under concurrency: whoever gets the pool, both callers compute
  // the reference bits (a serial caller keeps the backend's arithmetic)
  for i := 0 to 1 do
  begin
    a[i] := TAttentionCaller.Create(true);
    a[i].fBackend := fExact;
    a[i].fSeq := 20 + i * 5;
    a[i].fHeads := 4;
    a[i].fDim := 64;
    Fill(a[i].fQ, a[i].fSeq * 256, 800 + i, 1);
    Fill(a[i].fK, a[i].fSeq * 256, 810 + i, 1);
    Fill(a[i].fV, a[i].fSeq * 256, 820 + i, 2);
    SetLength(a[i].fKeep, a[i].fSeq);
    for j := 0 to a[i].fSeq - 1 do
      a[i].fKeep[j] := 1;
    SetLength(a[i].Expected, a[i].fSeq * 256);
    SetLength(sc, a[i].fSeq * a[i].fSeq);
    fRef.Attention(pointer(a[i].fQ), pointer(a[i].fK), pointer(a[i].fV), a[i].fSeq,
      4, 64, pointer(a[i].fKeep), pointer(sc), pointer(a[i].Expected));
  end;
  try
    for i := 0 to 1 do
      {$ifdef HASTTHREADSTART} a[i].Start; {$else} a[i].Resume; {$endif}
    for i := 0 to 1 do
      a[i].WaitFor;
    for i := 0 to 1 do
      CheckUtf8(not a[i].Mismatch and (a[i].Done = 40) and (a[i].FatalException = nil),
        'exact attention caller % = reference, every round done', [i]);
  finally
    for i := 0 to 1 do
      a[i].Free;
  end;
end;

procedure TTestInferFast.ForwardIsBitIdentical;
var
  dir, gguf: TFileName;
  m1, m2, m3: TXlmrModel;
  c1, c2, c3: TXlmrContext;
  ids: array[0..63] of integer;
  i: PtrInt;
  e1, e2, e3: TInferEmbedder;
  v1, v2, v3: TSingleDynArray;
  cmp: TInferCompare;
  e: double;
const
  TEXT = 'Kupferrohr 15 mm in Wand verlegen, inkl. Befestigung';
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  m1 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fRef);
  m2 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fExact);
  m3 := TXlmrModel.CreateFromGguf(dir + 'model.gguf', fFast);
  c1 := TXlmrContext.Create;
  c2 := TXlmrContext.Create;
  c3 := TXlmrContext.Create;
  try
    for i := 0 to 35 do
      ids[i] := 4 + (i * 37) mod 59;
    ids[0] := 0;
    ids[35] := 2;
    m1.Forward(c1, @ids, 1, 36);
    m2.Forward(c2, @ids, 1, 36);
    m3.Forward(c3, @ids, 1, 36);
    CheckUtf8(SameBits(pointer(c1.Hidden), pointer(c2.Hidden), c1.HiddenCount),
      'tiny GGUF, 36 tokens, exact mode');
    e := RelError(pointer(c3.Hidden), pointer(c1.Hidden), c1.HiddenCount);
    CheckUtf8(e <= 1e-3, 'tiny GGUF, fast mode: relative error %', [e]);
  finally
    c3.Free;
    c2.Free;
    c1.Free;
    m3.Free;
    m2.Free;
    m1.Free;
  end;
  gguf := FindBgeM3Gguf;
  if gguf = '' then
  begin
    AddConsole('SKIP BGE-M3 Q8 against the reference: no vendor/models/bge-m3-q8_0.gguf');
    exit;
  end;
  // one embedder at a time: three mappings of the 634 MB file would not fit
  // the address space of a 32-bit process
  e1 := TInferEmbedder.CreateFromGguf(gguf, fRef);
  try
    v1 := e1.Embed([TEXT]);
  finally
    e1.Free;
  end;
  e2 := TInferEmbedder.CreateFromGguf(gguf, fExact);
  try
    v2 := e2.Embed([TEXT]);
  finally
    e2.Free;
  end;
  e3 := TInferEmbedder.CreateFromGguf(gguf, fFast);
  try
    v3 := e3.Embed([TEXT]);
    CheckUtf8(SameBits(pointer(v1), pointer(v2), length(v1)),
      'BGE-M3 Q8 vector, exact mode = reference');
    cmp := InferCompare(pointer(v3), pointer(v1), length(v1), 1, 1);
    // float lanes against double: the activation quantization amplifies the
    // rounding differences (measured 0.99981); the gate against lembed is the
    // corpus test in test.infer.gguf
    CheckUtf8(cmp.Cosine >= 0.9995, 'BGE-M3 Q8 vector, fast mode: cosine %',
      [cmp.Cosine]);
  finally
    e3.Free;
  end;
end;

procedure TTestInferFast.NativeEmbedderAdapter;
var
  dir: TFileName;
  a, b: IEmbedder;
  v: TLlmEmbedding;
  vs: TLlmEmbeddingDynArray;
  texts: TRawUtf8DynArray;
  long: RawUtf8;
  i: PtrInt;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-gguf');
  if CheckFailed(dir <> '', 'tiny-gguf found') then
    exit;
  a := SharedNativeEmbedder(StringToUtf8(dir + 'model.gguf'), 'tiny-test', 2);
  b := SharedNativeEmbedder('ignored once cached', 'tiny-test');
  CheckUtf8(a = b, 'one instance per model name');
  CheckEqual(a.Model, 'tiny-test', 'model name');
  v := a.Embed('the abc');
  CheckEqual(length(v), 32, 'tiny dimension');
  SetLength(texts, 3);
  texts[0] := 'abc';
  texts[1] := 'the abc';
  texts[2] := 'in the end';
  vs := a.EmbedBatch(texts);
  CheckEqual(length(vs), 3, 'one vector per text');
  CheckUtf8(CompareMem(pointer(vs[1]), pointer(v), 32 * SizeOf(single)),
    'EmbedBatch = Embed');
  long := '';
  for i := 1 to 40 do
    long := long + 'the ';
  refused := false;
  try
    a.Embed(long);
  except
    on EInferError do
      refused := true;
  end;
  CheckUtf8(refused, 'too many tokens refused, never cut');
end;

end.
