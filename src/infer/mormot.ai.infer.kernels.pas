/// native inference engine - CPU-specific kernels of the fast backend
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - one job only so far: the int32 dot product of Q8_0 block pairs, the inner
//   loop of every quantized matrix product. Integer sums are exact, so every
//   kernel returns the same numbers as the plain Pascal loop - the fast backend
//   stays bit-identical to the reference
// - the 4-row variant multiplies one activation block with four weight rows:
//   the activations come from L2, the weight tile from L1, and loading each
//   activation block once per four rows instead of once per row is what lets
//   more than three cores share one L2 (Snapdragon X: 4 cores per 12 MB L2)
// - aarch64 with the dot product extension (ARMv8.2 SDOT, detected at runtime):
//   NEON; FPC 3.2.x cannot assemble NEON register lists, so the vector
//   instructions are embedded as their encodings, each commented with its
//   mnemonic. Everything else: the Pascal loop
unit mormot.ai.infer.kernels;

{
  *****************************************************************************

    - InferQ8BlockDots: int32 dot of Q8_0 block pairs, dispatched per CPU
    - INFER_HALF: every FP16 value as single, for the scale lookups

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.ai.infer.tensor;

type
  /// aSums[b] := sum over i of w[b].qs[i] * x[b].qs[i], for b in 0..aBlocks-1
  TInferQ8BlockDots = procedure(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
    aSums: PIntegerArray);
  /// for the four weight rows w + k aStride (k = 0..3, aStride in bytes)
  // against one activation row x with block scales aDx (as double):
  // aAcc[k] := aAcc[k] + sum over b of int(block b) * (scale(w block) * aDx[b]),
  // block by block in order - the exact arithmetic of the reference Linear
  TInferQ8Linear4 = procedure(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
    aDx: PDoubleArray; aStride: PtrInt; aAcc: PDoubleArray);
  /// quantize aBlocks blocks of 32 singles (see InferQ8QuantizeRow)
  TInferQ8QuantizeRow = function(x: PSingle; aBlocks: PtrInt;
    y: PInferBlockQ8_0): PtrInt;
  /// the float row kernels of the fast backend's attention (see InferDotsF32)
  TInferDotsF32 = procedure(a, b: PSingle; aStride, aCount, aDim: PtrInt;
    aOut: PSingleArray);
  /// aOut[k] / aOut[4 + k] := the dot product of weight row w + k aStride with
  // activation row xA / xB, in float lanes: no bias, not the double reference
  TInferQ8DotF32x4x2 = procedure(w, xA, xB: PInferBlockQ8_0; aBlocks: PtrInt;
    aDxA, aDxB: PSingleArray; aStride: PtrInt; aOut: PSingleArray);
  /// the same for the four weight rows w, w + aStride, w + 2 aStride and
  // w + 3 aStride (aStride in bytes) against one row x:
  // aSums[b * 4 + k] = block b of row k
  TInferQ8BlockDots4 = procedure(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
    aSums: PIntegerArray; aStride: PtrInt);

var
  /// the Q8_0 block dot kernel of this CPU (set at unit initialization)
  InferQ8BlockDots: TInferQ8BlockDots;
  /// the 4-row Q8_0 block dot kernel of this CPU
  InferQ8BlockDots4: TInferQ8BlockDots4;
  /// the fused 4-row Q8_0 Linear kernel of this CPU, nil without one: the
  // caller then combines InferQ8BlockDots4 with its own scaling loop
  InferQ8Linear4: TInferQ8Linear4;
  /// the fused kernel for four weight rows against two activation rows x and
  // x + aStride (scales aDx and aDx + aBlocks); aAcc is [row A: 4, row B: 4];
  // nil without one
  InferQ8Linear4x2: TInferQ8Linear4;
  /// the FP32 kernel of the fast backend, nil without one: four weight rows
  // against two activation rows, float lanes (see Q8DotF32x4x2Pascal)
  InferQ8DotF32x4x2: TInferQ8DotF32x4x2;
  /// float attention scores of the fast backend, nil without a kernel:
  // aOut[j] := dot(q, k + j * aStride bytes) over aDim (a multiple of 16)
  InferDotsF32: TInferDotsF32;
  /// the activation quantizer of this CPU, nil without one: same bytes as
  // InferQuantizeRowQ8_0; returns 0, or 1 + the first block with a NaN or Inf
  InferQ8QuantizeRow: TInferQ8QuantizeRow;
  /// float weighted value sum of the fast backend, nil without a kernel:
  // y += sum over j of p[j] * (v + j * aStride bytes), aDim a multiple of 16
  InferAxpyRowsF32: TInferDotsF32;
  /// which kernel InferQ8BlockDots is: 'pascal', 'neon-dotprod'
  InferQ8KernelName: RawUtf8;
  /// every IEEE half (raw bits) as single - 256 KB, filled at initialization
  INFER_HALF: array[word] of single;

/// the plain Pascal kernel - the yardstick of every other one
procedure InferQ8BlockDotsPascal(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray);

/// the plain Pascal 4-row kernel
procedure InferQ8BlockDots4Pascal(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray; aStride: PtrInt);


implementation

procedure InferQ8BlockDotsPascal(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray);
var
  b, j: PtrInt;
  sum: integer;
begin
  for b := 0 to aBlocks - 1 do
  begin
    sum := 0; // at most 32 * 128 * 128: no overflow
    for j := 0 to 31 do
      inc(sum, integer(w^.qs[j]) * x^.qs[j]);
    aSums^[b] := sum;
    inc(w);
    inc(x);
  end;
end;

procedure InferQ8BlockDots4Pascal(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray; aStride: PtrInt);
var
  b, k, j: PtrInt;
  sum: integer;
  wk, xb: PInferBlockQ8_0;
begin
  xb := x;
  for b := 0 to aBlocks - 1 do
  begin
    for k := 0 to 3 do
    begin
      wk := pointer(PAnsiChar(w) + k * aStride + b * SizeOf(TInferBlockQ8_0));
      sum := 0;
      for j := 0 to 31 do
        inc(sum, integer(wk^.qs[j]) * xb^.qs[j]);
      aSums^[b * 4 + k] := sum;
    end;
    inc(xb);
  end;
end;

{$ifdef CPUAARCH64}
{$ifdef FPC}

/// NEON with SDOT: two 16-byte loads per side, two SDOT, one horizontal add
procedure Q8BlockDotsNeonDot(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray); assembler; nostackframe;
asm
  // x0 = w, x1 = x, x2 = aBlocks, x3 = aSums; the 32 qs follow the 2-byte scale
  cbz x2, .Ldone
.Lloop:
  .long 0x3cc02000 // ldur q0, [x0, #2]
  .long 0x3cc12001 // ldur q1, [x0, #18]
  .long 0x3cc02022 // ldur q2, [x1, #2]
  .long 0x3cc12023 // ldur q3, [x1, #18]
  .long 0x4f000404 // movi v4.4s, #0
  .long 0x4e829404 // sdot v4.4s, v0.16b, v2.16b
  .long 0x4e839424 // sdot v4.4s, v1.16b, v3.16b
  .long 0x4eb1b884 // addv s4, v4.4s
  .long 0xbc004464 // str s4, [x3], #4
  add x0, x0, #34
  add x1, x1, #34
  subs x2, x2, #1
  b.ne .Lloop
.Ldone:
end;

/// four weight rows against one activation row: 8 SDOT per block, one load of
// the activation block; v16..v26 only (v8..v15 are callee-saved)
procedure Q8BlockDots4NeonDot(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aSums: PIntegerArray; aStride: PtrInt); assembler; nostackframe;
asm
  // x0 = w row 0, x1 = x, x2 = aBlocks, x3 = aSums, x4 = aStride
  cbz x2, .Ldone4
  add x5, x0, x4
  add x6, x5, x4
  add x7, x6, x4
.Lloop4:
  .long 0x3cc02020 // ldur q0, [x1, #2]
  .long 0x3cc12021 // ldur q1, [x1, #18]
  .long 0x3cc02002 // ldur q2, [x0, #2]
  .long 0x3cc12003 // ldur q3, [x0, #18]
  .long 0x3cc020a5 // ldur q5, [x5, #2]
  .long 0x3cc120a6 // ldur q6, [x5, #18]
  .long 0x3cc020d0 // ldur q16, [x6, #2]
  .long 0x3cc120d1 // ldur q17, [x6, #18]
  .long 0x3cc020f2 // ldur q18, [x7, #2]
  .long 0x3cc120f3 // ldur q19, [x7, #18]
  .long 0x4f000414 // movi v20.4s, #0
  .long 0x4f000415 // movi v21.4s, #0
  .long 0x4f000416 // movi v22.4s, #0
  .long 0x4f000417 // movi v23.4s, #0
  .long 0x4e809454 // sdot v20.4s, v2.16b, v0.16b
  .long 0x4e8094b5 // sdot v21.4s, v5.16b, v0.16b
  .long 0x4e809616 // sdot v22.4s, v16.16b, v0.16b
  .long 0x4e809657 // sdot v23.4s, v18.16b, v0.16b
  .long 0x4e819474 // sdot v20.4s, v3.16b, v1.16b
  .long 0x4e8194d5 // sdot v21.4s, v6.16b, v1.16b
  .long 0x4e819636 // sdot v22.4s, v17.16b, v1.16b
  .long 0x4e819677 // sdot v23.4s, v19.16b, v1.16b
  .long 0x4eb5be98 // addp v24.4s, v20.4s, v21.4s
  .long 0x4eb7bed9 // addp v25.4s, v22.4s, v23.4s
  .long 0x4eb9bf1a // addp v26.4s, v24.4s, v25.4s - the four row sums
  .long 0x3c81047a // str q26, [x3], #16
  add x0, x0, #34
  add x1, x1, #34
  add x5, x5, #34
  add x6, x6, #34
  add x7, x7, #34
  subs x2, x2, #1
  b.ne .Lloop4
.Ldone4:
end;

/// fused Q8_0 x Q8_0 for four weight rows against one activation row, the whole
// Linear inner loop of TInferCpuFastBackend: per block the int32 sums (SDOT), then
// in float64 lanes acc[k] := acc[k] + sum[k] * (dw[k] * dx[b]) - lane-wise
// IEEE double, separate multiply and add (no FMA), so each row accumulates
// exactly as the Pascal reference does; aAcc holds the 4 start values (bias)
// and receives the 4 results
procedure Q8Linear4NeonDot(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aDx: PDoubleArray; aStride: PtrInt; aAcc: PDoubleArray); assembler; nostackframe;
asm
  // x0 = w row 0, x1 = x, x2 = aBlocks, x3 = aDx, x4 = aStride, x5 = aAcc
  cbz x2, .Ldonel4
  add x6, x0, x4
  add x7, x6, x4
  add x9, x7, x4
  .long 0xad407cbe // ldp q30, q31, [x5]
.Lloopl4:
  .long 0x7d400012 // ldr h18, [x0]
  .long 0x7d4000d3 // ldr h19, [x6]
  .long 0x7d4000f4 // ldr h20, [x7]
  .long 0x7d400135 // ldr h21, [x9]
  .long 0x1ee2c252 // fcvt d18, h18
  .long 0x1ee2c273 // fcvt d19, h19
  .long 0x1ee2c294 // fcvt d20, h20
  .long 0x1ee2c2b5 // fcvt d21, h21
  .long 0x6e08065b // mov v27.d[0], v18.d[0]
  .long 0x6e18067b // mov v27.d[1], v19.d[0]
  .long 0x6e08069c // mov v28.d[0], v20.d[0]
  .long 0x6e1806bc // mov v28.d[1], v21.d[0]
  .long 0x4ddfcc7d // ld1r {v29.2d}, [x3], #8
  .long 0x6e7ddf7b // fmul v27.2d, v27.2d, v29.2d
  .long 0x6e7ddf9c // fmul v28.2d, v28.2d, v29.2d
  .long 0x3cc02020 // ldur q0, [x1, #2]
  .long 0x3cc12021 // ldur q1, [x1, #18]
  .long 0x3cc02002 // ldur q2, [x0, #2]
  .long 0x3cc12003 // ldur q3, [x0, #18]
  .long 0x3cc020c4 // ldur q4, [x6, #2]
  .long 0x3cc120c5 // ldur q5, [x6, #18]
  .long 0x3cc020e6 // ldur q6, [x7, #2]
  .long 0x3cc120e7 // ldur q7, [x7, #18]
  .long 0x3cc02130 // ldur q16, [x9, #2]
  .long 0x3cc12131 // ldur q17, [x9, #18]
  .long 0x4f000412 // movi v18.4s, #0x0
  .long 0x4f000413 // movi v19.4s, #0x0
  .long 0x4f000414 // movi v20.4s, #0x0
  .long 0x4f000415 // movi v21.4s, #0x0
  .long 0x4e809452 // sdot v18.4s, v2.16b, v0.16b
  .long 0x4e809493 // sdot v19.4s, v4.16b, v0.16b
  .long 0x4e8094d4 // sdot v20.4s, v6.16b, v0.16b
  .long 0x4e809615 // sdot v21.4s, v16.16b, v0.16b
  .long 0x4e819472 // sdot v18.4s, v3.16b, v1.16b
  .long 0x4e8194b3 // sdot v19.4s, v5.16b, v1.16b
  .long 0x4e8194f4 // sdot v20.4s, v7.16b, v1.16b
  .long 0x4e819635 // sdot v21.4s, v17.16b, v1.16b
  .long 0x4eb3be56 // addp v22.4s, v18.4s, v19.4s
  .long 0x4eb5be97 // addp v23.4s, v20.4s, v21.4s
  .long 0x4eb7bed8 // addp v24.4s, v22.4s, v23.4s
  .long 0x0f20a719 // sxtl v25.2d, v24.2s
  .long 0x4f20a71a // sxtl2 v26.2d, v24.4s
  .long 0x4e61db39 // scvtf v25.2d, v25.2d
  .long 0x4e61db5a // scvtf v26.2d, v26.2d
  .long 0x6e7bdf39 // fmul v25.2d, v25.2d, v27.2d
  .long 0x6e7cdf5a // fmul v26.2d, v26.2d, v28.2d
  .long 0x4e79d7de // fadd v30.2d, v30.2d, v25.2d
  .long 0x4e7ad7ff // fadd v31.2d, v31.2d, v26.2d
  add x0, x0, #34
  add x1, x1, #34
  add x6, x6, #34
  add x7, x7, #34
  add x9, x9, #34
  subs x2, x2, #1
  b.ne .Lloopl4
  .long 0xad007cbe // stp q30, q31, [x5]
.Ldonel4:
end;

/// fused Q8_0 x Q8_0 for four weight rows against TWO activation rows (x and
// x + aStride, scales aDx and aDx + aBlocks): each weight block and its scale
// are loaded once for both rows. Same per-row arithmetic as Q8Linear4NeonDot;
// aAcc holds [row A: 4, row B: 4] start values and receives the results.
// Uses v8..v15, whose low halves are callee-saved: d8..d15 kept on the stack
procedure Q8Linear4x2NeonDot(w, x: PInferBlockQ8_0; aBlocks: PtrInt;
  aDx: PDoubleArray; aStride: PtrInt; aAcc: PDoubleArray); assembler; nostackframe;
asm
  // x0 = w row 0, x1 = x row A, x2 = aBlocks, x3 = aDx row A, x4 = aStride,
  // x5 = aAcc; x6/x7/x9 = w rows 1..3, x11 = x row B, x10 = aDx row B
  cbz x2, .Ldone42
  add x6, x0, x4
  add x7, x6, x4
  add x9, x7, x4
  add x11, x1, x4
  .long 0x8b020c6a // add x10, x3, x2, lsl #3
  .long 0x6dbc27e8 // stp d8, d9, [sp, #-64]!
  .long 0x6d012fea // stp d10, d11, [sp, #16]
  .long 0x6d0237ec // stp d12, d13, [sp, #32]
  .long 0x6d033fee // stp d14, d15, [sp, #48]
  .long 0xad4074bc // ldp q28, q29, [x5]
  .long 0xad417cbe // ldp q30, q31, [x5, #32]
.Lloop42:
  .long 0x7d400014 // ldr h20, [x0]
  .long 0x7d4000d5 // ldr h21, [x6]
  .long 0x7d4000f6 // ldr h22, [x7]
  .long 0x7d400137 // ldr h23, [x9]
  .long 0x1ee2c294 // fcvt d20, h20
  .long 0x1ee2c2b5 // fcvt d21, h21
  .long 0x1ee2c2d6 // fcvt d22, h22
  .long 0x1ee2c2f7 // fcvt d23, h23
  .long 0x6e08068e // mov v14.d[0], v20.d[0]
  .long 0x6e1806ae // mov v14.d[1], v21.d[0]
  .long 0x6e0806cf // mov v15.d[0], v22.d[0]
  .long 0x6e1806ef // mov v15.d[1], v23.d[0]
  .long 0x3cc02020 // ldur q0, [x1, #2]
  .long 0x3cc12021 // ldur q1, [x1, #18]
  .long 0x3cc02162 // ldur q2, [x11, #2]
  .long 0x3cc12163 // ldur q3, [x11, #18]
  .long 0x3cc02004 // ldur q4, [x0, #2]
  .long 0x3cc12005 // ldur q5, [x0, #18]
  .long 0x3cc020c6 // ldur q6, [x6, #2]
  .long 0x3cc120c7 // ldur q7, [x6, #18]
  .long 0x3cc020f0 // ldur q16, [x7, #2]
  .long 0x3cc120f1 // ldur q17, [x7, #18]
  .long 0x3cc02132 // ldur q18, [x9, #2]
  .long 0x3cc12133 // ldur q19, [x9, #18]
  .long 0x4f000414 // movi v20.4s, #0x0
  .long 0x4f000415 // movi v21.4s, #0x0
  .long 0x4f000416 // movi v22.4s, #0x0
  .long 0x4f000417 // movi v23.4s, #0x0
  .long 0x4f000418 // movi v24.4s, #0x0
  .long 0x4f000419 // movi v25.4s, #0x0
  .long 0x4f00041a // movi v26.4s, #0x0
  .long 0x4f00041b // movi v27.4s, #0x0
  .long 0x4e809494 // sdot v20.4s, v4.16b, v0.16b
  .long 0x4e8094d5 // sdot v21.4s, v6.16b, v0.16b
  .long 0x4e809616 // sdot v22.4s, v16.16b, v0.16b
  .long 0x4e809657 // sdot v23.4s, v18.16b, v0.16b
  .long 0x4e829498 // sdot v24.4s, v4.16b, v2.16b
  .long 0x4e8294d9 // sdot v25.4s, v6.16b, v2.16b
  .long 0x4e82961a // sdot v26.4s, v16.16b, v2.16b
  .long 0x4e82965b // sdot v27.4s, v18.16b, v2.16b
  .long 0x4e8194b4 // sdot v20.4s, v5.16b, v1.16b
  .long 0x4e8194f5 // sdot v21.4s, v7.16b, v1.16b
  .long 0x4e819636 // sdot v22.4s, v17.16b, v1.16b
  .long 0x4e819677 // sdot v23.4s, v19.16b, v1.16b
  .long 0x4e8394b8 // sdot v24.4s, v5.16b, v3.16b
  .long 0x4e8394f9 // sdot v25.4s, v7.16b, v3.16b
  .long 0x4e83963a // sdot v26.4s, v17.16b, v3.16b
  .long 0x4e83967b // sdot v27.4s, v19.16b, v3.16b
  .long 0x4eb5be88 // addp v8.4s, v20.4s, v21.4s
  .long 0x4eb7bec9 // addp v9.4s, v22.4s, v23.4s
  .long 0x4ea9bd0a // addp v10.4s, v8.4s, v9.4s
  .long 0x4eb9bf0b // addp v11.4s, v24.4s, v25.4s
  .long 0x4ebbbf4c // addp v12.4s, v26.4s, v27.4s
  .long 0x4eacbd6d // addp v13.4s, v11.4s, v12.4s
  .long 0x0f20a554 // sxtl v20.2d, v10.2s
  .long 0x4f20a555 // sxtl2 v21.2d, v10.4s
  .long 0x0f20a5b6 // sxtl v22.2d, v13.2s
  .long 0x4f20a5b7 // sxtl2 v23.2d, v13.4s
  .long 0x4e61da94 // scvtf v20.2d, v20.2d
  .long 0x4e61dab5 // scvtf v21.2d, v21.2d
  .long 0x4e61dad6 // scvtf v22.2d, v22.2d
  .long 0x4e61daf7 // scvtf v23.2d, v23.2d
  .long 0x4ddfcc78 // ld1r {v24.2d}, [x3], #8
  .long 0x4ddfcd59 // ld1r {v25.2d}, [x10], #8
  .long 0x6e78ddda // fmul v26.2d, v14.2d, v24.2d
  .long 0x6e78ddfb // fmul v27.2d, v15.2d, v24.2d
  .long 0x6e7ade94 // fmul v20.2d, v20.2d, v26.2d
  .long 0x6e7bdeb5 // fmul v21.2d, v21.2d, v27.2d
  .long 0x6e79ddda // fmul v26.2d, v14.2d, v25.2d
  .long 0x6e79ddfb // fmul v27.2d, v15.2d, v25.2d
  .long 0x6e7aded6 // fmul v22.2d, v22.2d, v26.2d
  .long 0x6e7bdef7 // fmul v23.2d, v23.2d, v27.2d
  .long 0x4e74d79c // fadd v28.2d, v28.2d, v20.2d
  .long 0x4e75d7bd // fadd v29.2d, v29.2d, v21.2d
  .long 0x4e76d7de // fadd v30.2d, v30.2d, v22.2d
  .long 0x4e77d7ff // fadd v31.2d, v31.2d, v23.2d
  add x0, x0, #34
  add x1, x1, #34
  add x6, x6, #34
  add x7, x7, #34
  add x9, x9, #34
  add x11, x11, #34
  subs x2, x2, #1
  b.ne .Lloop42
  .long 0xad0074bc // stp q28, q29, [x5]
  .long 0xad017cbe // stp q30, q31, [x5, #32]
  .long 0x6d433fee // ldp d14, d15, [sp, #48]
  .long 0x6d4237ec // ldp d12, d13, [sp, #32]
  .long 0x6d412fea // ldp d10, d11, [sp, #16]
  .long 0x6cc427e8 // ldp d8, d9, [sp], #64
.Ldone42:
end;

/// FP32 Q8_0 x Q8_0 for four weight rows against two activation rows (the fast
// arithmetic, ggml's way): per block the exact int32 SDOT lane sums are
// converted to float and multiply-added (FMA) by the float scale dw * dx into
// four float lanes per output; the lanes are added (pairwise) only at the end.
// aOut receives [row A: 4, row B: 4] dot products without bias. Not the double
// reference - checked against it within a tolerance. For one activation row
// pass xB = xA. Uses v8..v15: d8..d15 kept on the stack
procedure Q8DotF32x4x2NeonDot(w, xA, xB: PInferBlockQ8_0; aBlocks: PtrInt;
  aDxA, aDxB: PSingleArray; aStride: PtrInt; aOut: PSingleArray); assembler; nostackframe;
asm
  // x0 = w row 0, x1 = xA, x2 = xB, x3 = aBlocks, x4 = aDxA, x5 = aDxB,
  // x6 = aStride, x7 = aOut; x9/x10/x11 = w rows 1..3
  add x9, x0, x6
  add x10, x9, x6
  add x11, x10, x6
  .long 0x6dbc27e8 // stp d8, d9, [sp, #-64]!
  .long 0x6d012fea // stp d10, d11, [sp, #16]
  .long 0x6d0237ec // stp d12, d13, [sp, #32]
  .long 0x6d033fee // stp d14, d15, [sp, #48]
  .long 0x4f000418 // movi v24.4s, #0x0
  .long 0x4f000419 // movi v25.4s, #0x0
  .long 0x4f00041a // movi v26.4s, #0x0
  .long 0x4f00041b // movi v27.4s, #0x0
  .long 0x4f00041c // movi v28.4s, #0x0
  .long 0x4f00041d // movi v29.4s, #0x0
  .long 0x4f00041e // movi v30.4s, #0x0
  .long 0x4f00041f // movi v31.4s, #0x0
  cbz x3, .Lendf42
.Lloopf42:
  .long 0x0d404014 // ld1 {v20.h}[0], [x0]
  .long 0x0d404934 // ld1 {v20.h}[1], [x9]
  .long 0x0d405154 // ld1 {v20.h}[2], [x10]
  .long 0x0d405974 // ld1 {v20.h}[3], [x11]
  .long 0x0e217a94 // fcvtl v20.4s, v20.4h
  .long 0x4ddfc895 // ld1r {v21.4s}, [x4], #4
  .long 0x4ddfc8b6 // ld1r {v22.4s}, [x5], #4
  .long 0x6e35de95 // fmul v21.4s, v20.4s, v21.4s
  .long 0x6e36de96 // fmul v22.4s, v20.4s, v22.4s
  .long 0x3cc02020 // ldur q0, [x1, #2]
  .long 0x3cc12021 // ldur q1, [x1, #18]
  .long 0x3cc02042 // ldur q2, [x2, #2]
  .long 0x3cc12043 // ldur q3, [x2, #18]
  .long 0x3cc02004 // ldur q4, [x0, #2]
  .long 0x3cc12005 // ldur q5, [x0, #18]
  .long 0x3cc02126 // ldur q6, [x9, #2]
  .long 0x3cc12127 // ldur q7, [x9, #18]
  .long 0x3cc02150 // ldur q16, [x10, #2]
  .long 0x3cc12151 // ldur q17, [x10, #18]
  .long 0x3cc02172 // ldur q18, [x11, #2]
  .long 0x3cc12173 // ldur q19, [x11, #18]
  .long 0x4f000408 // movi v8.4s, #0x0
  .long 0x4f000409 // movi v9.4s, #0x0
  .long 0x4f00040a // movi v10.4s, #0x0
  .long 0x4f00040b // movi v11.4s, #0x0
  .long 0x4f00040c // movi v12.4s, #0x0
  .long 0x4f00040d // movi v13.4s, #0x0
  .long 0x4f00040e // movi v14.4s, #0x0
  .long 0x4f00040f // movi v15.4s, #0x0
  .long 0x4e809488 // sdot v8.4s, v4.16b, v0.16b
  .long 0x4e8094c9 // sdot v9.4s, v6.16b, v0.16b
  .long 0x4e80960a // sdot v10.4s, v16.16b, v0.16b
  .long 0x4e80964b // sdot v11.4s, v18.16b, v0.16b
  .long 0x4e82948c // sdot v12.4s, v4.16b, v2.16b
  .long 0x4e8294cd // sdot v13.4s, v6.16b, v2.16b
  .long 0x4e82960e // sdot v14.4s, v16.16b, v2.16b
  .long 0x4e82964f // sdot v15.4s, v18.16b, v2.16b
  .long 0x4e8194a8 // sdot v8.4s, v5.16b, v1.16b
  .long 0x4e8194e9 // sdot v9.4s, v7.16b, v1.16b
  .long 0x4e81962a // sdot v10.4s, v17.16b, v1.16b
  .long 0x4e81966b // sdot v11.4s, v19.16b, v1.16b
  .long 0x4e8394ac // sdot v12.4s, v5.16b, v3.16b
  .long 0x4e8394ed // sdot v13.4s, v7.16b, v3.16b
  .long 0x4e83962e // sdot v14.4s, v17.16b, v3.16b
  .long 0x4e83966f // sdot v15.4s, v19.16b, v3.16b
  .long 0x4e21d908 // scvtf v8.4s, v8.4s
  .long 0x4e21d929 // scvtf v9.4s, v9.4s
  .long 0x4e21d94a // scvtf v10.4s, v10.4s
  .long 0x4e21d96b // scvtf v11.4s, v11.4s
  .long 0x4e21d98c // scvtf v12.4s, v12.4s
  .long 0x4e21d9ad // scvtf v13.4s, v13.4s
  .long 0x4e21d9ce // scvtf v14.4s, v14.4s
  .long 0x4e21d9ef // scvtf v15.4s, v15.4s
  .long 0x4f951118 // fmla v24.4s, v8.4s, v21.s[0]
  .long 0x4fb51139 // fmla v25.4s, v9.4s, v21.s[1]
  .long 0x4f95195a // fmla v26.4s, v10.4s, v21.s[2]
  .long 0x4fb5197b // fmla v27.4s, v11.4s, v21.s[3]
  .long 0x4f96119c // fmla v28.4s, v12.4s, v22.s[0]
  .long 0x4fb611bd // fmla v29.4s, v13.4s, v22.s[1]
  .long 0x4f9619de // fmla v30.4s, v14.4s, v22.s[2]
  .long 0x4fb619ff // fmla v31.4s, v15.4s, v22.s[3]
  add x0, x0, #34
  add x1, x1, #34
  add x2, x2, #34
  add x9, x9, #34
  add x10, x10, #34
  add x11, x11, #34
  subs x3, x3, #1
  b.ne .Lloopf42
.Lendf42:
  .long 0x6e39d718 // faddp v24.4s, v24.4s, v25.4s
  .long 0x6e3bd75a // faddp v26.4s, v26.4s, v27.4s
  .long 0x6e3ad718 // faddp v24.4s, v24.4s, v26.4s
  .long 0x6e3dd79c // faddp v28.4s, v28.4s, v29.4s
  .long 0x6e3fd7de // faddp v30.4s, v30.4s, v31.4s
  .long 0x6e3ed79c // faddp v28.4s, v28.4s, v30.4s
  .long 0xad0070f8 // stp q24, q28, [x7]
  .long 0x6d433fee // ldp d14, d15, [sp, #48]
  .long 0x6d4237ec // ldp d12, d13, [sp, #32]
  .long 0x6d412fea // ldp d10, d11, [sp, #16]
  .long 0x6cc427e8 // ldp d8, d9, [sp], #64
end;

/// aOut[j] := dot(q, k + j * aStride) over aDim singles (aDim a multiple of 16),
// for j in 0 .. aKeys - 1: four float lanes per 16 values, FMA, pairwise sum
procedure DotsF32Neon(q, k: PSingle; aStride, aKeys, aDim: PtrInt;
  aOut: PSingleArray); assembler; nostackframe;
asm
  // x0 = q, x1 = k, x2 = aStride (bytes), x3 = aKeys, x4 = aDim, x5 = aOut
  cbz x3, .Ldonedots
.Lkeydots:
  mov x6, x1
  mov x7, x0
  mov x8, x4
  .long 0x4f000400 // movi v0.4s, #0
  .long 0x4f000401 // movi v1.4s, #0
  .long 0x4f000402 // movi v2.4s, #0
  .long 0x4f000403 // movi v3.4s, #0
.Ldimdots:
  .long 0x4cdf28e4 // ld1 {v4.4s-v7.4s}, [x7], #64
  .long 0x4cdf28d0 // ld1 {v16.4s-v19.4s}, [x6], #64
  .long 0x4e30cc80 // fmla v0.4s, v4.4s, v16.4s
  .long 0x4e31cca1 // fmla v1.4s, v5.4s, v17.4s
  .long 0x4e32ccc2 // fmla v2.4s, v6.4s, v18.4s
  .long 0x4e33cce3 // fmla v3.4s, v7.4s, v19.4s
  subs x8, x8, #16
  b.ne .Ldimdots
  .long 0x4e21d400 // fadd v0.4s, v0.4s, v1.4s
  .long 0x4e23d442 // fadd v2.4s, v2.4s, v3.4s
  .long 0x4e22d400 // fadd v0.4s, v0.4s, v2.4s
  .long 0x6e20d400 // faddp v0.4s, v0.4s, v0.4s
  .long 0x7e30d800 // faddp s0, v0.2s
  .long 0xbc0044a0 // str s0, [x5], #4
  add x1, x1, x2
  subs x3, x3, #1
  b.ne .Lkeydots
.Ldonedots:
end;

/// y[0 .. aDim) += sum over j of p[j] * (v + j * aStride)[0 .. aDim), aDim a
// multiple of 16, j in order; y stays in L1 between the keys
procedure AxpyRowsF32Neon(p, v: PSingle; aStride, aKeys, aDim: PtrInt;
  y: PSingleArray); assembler; nostackframe;
asm
  // x0 = p, x1 = v, x2 = aStride (bytes), x3 = aKeys, x4 = aDim, x5 = y
  cbz x3, .Ldoneaxpy
.Lkeyaxpy:
  .long 0x4ddfc81f // ld1r {v31.4s}, [x0], #4
  mov x6, x1
  mov x7, x5
  mov x8, x4
.Ldimaxpy:
  .long 0x4cdf28c4 // ld1 {v4.4s-v7.4s}, [x6], #64
  .long 0x4c4028f0 // ld1 {v16.4s-v19.4s}, [x7]
  .long 0x4e3fcc90 // fmla v16.4s, v4.4s, v31.4s
  .long 0x4e3fccb1 // fmla v17.4s, v5.4s, v31.4s
  .long 0x4e3fccd2 // fmla v18.4s, v6.4s, v31.4s
  .long 0x4e3fccf3 // fmla v19.4s, v7.4s, v31.4s
  .long 0x4c9f28f0 // st1 {v16.4s-v19.4s}, [x7], #64
  subs x8, x8, #16
  b.ne .Ldimaxpy
  add x1, x1, x2
  subs x3, x3, #1
  b.ne .Lkeyaxpy
.Ldoneaxpy:
end;

/// quantize aBlocks blocks of 32 singles at x into Q8_0 blocks at y, exactly
// as InferQuantizeRowQ8_0: d = max|x| / 127 (single, kept as FP16 rounded to
// nearest even), q = x * (1 / d) rounded to nearest even (FCVTNS, as ggml's NEON
// path), all q = 0 when d <= 2^-128; returns 0, or 1 + the index of the first
// block holding a NaN or an infinity (nothing is written for it or after it)
function Q8QuantizeRowNeon(x: PSingle; aBlocks: PtrInt;
  y: PInferBlockQ8_0): PtrInt; assembler; nostackframe;
asm
  // x0 = x, x1 = aBlocks, x2 = y; x11 = aBlocks for the status
  mov x11, x1
  cbz x1, .Lokq
  .long 0x52a85fc9 // mov w9, #0x42fe0000 // #1123942400
  .long 0x1e270139 // fmov s25, w9
  .long 0x12b01009 // mov w9, #0x7f7fffff // #2139095039
  .long 0x1e270138 // fmov s24, w9
  .long 0x52a00409 // mov w9, #0x200000 // #2097152
  .long 0x1e27013a // fmov s26, w9
  .long 0x1e2e101b // fmov s27, #1.000000000000000000e+00
  .long 0x529e0009 // mov w9, #0xf000
  .long 0x72a8efe9 // movk w9, #0x477f, lsl #16
  .long 0x1e27013d // fmov s29, w9 (65520.0)
.Lloopq:
  .long 0x4cdf2800 // ld1 {v0.4s-v3.4s}, [x0], #64
  .long 0x4cdf2804 // ld1 {v4.4s-v7.4s}, [x0], #64
  .long 0x4ea0f810 // fabs v16.4s, v0.4s
  .long 0x4ea0f831 // fabs v17.4s, v1.4s
  .long 0x4ea0f852 // fabs v18.4s, v2.4s
  .long 0x4ea0f873 // fabs v19.4s, v3.4s
  .long 0x4ea0f894 // fabs v20.4s, v4.4s
  .long 0x4ea0f8b5 // fabs v21.4s, v5.4s
  .long 0x4ea0f8d6 // fabs v22.4s, v6.4s
  .long 0x4ea0f8f7 // fabs v23.4s, v7.4s
  .long 0x4e31f610 // fmax v16.4s, v16.4s, v17.4s
  .long 0x4e33f652 // fmax v18.4s, v18.4s, v19.4s
  .long 0x4e35f694 // fmax v20.4s, v20.4s, v21.4s
  .long 0x4e37f6d6 // fmax v22.4s, v22.4s, v23.4s
  .long 0x4e32f610 // fmax v16.4s, v16.4s, v18.4s
  .long 0x4e36f694 // fmax v20.4s, v20.4s, v22.4s
  .long 0x4e34f610 // fmax v16.4s, v16.4s, v20.4s
  .long 0x6e30fa10 // fmaxv s16, v16.4s
  .long 0x1e382200 // fcmp s16, s24
  b.vs .Lbadq
  b.gt .Lbadq
  .long 0x1e391a11 // fdiv s17, s16, s25
  // a scale beyond the largest half: +Inf, written as bits - fcvt would
  // raise the overflow trap FPC enables (InferSingleToHalf gives the same)
  .long 0x1e3d2220 // fcmp s17, s29 (65520: rounds to +Inf in FP16)
  b.ge .Lhinfq
  .long 0x1e23c233 // fcvt h19, s17
  .long 0x7d000053 // str h19, [x2]
  b .Lhdoneq
.Lhinfq:
  .long 0x528f8009 // mov w9, #0x7c00
  .long 0x79000049 // strh w9, [x2]
.Lhdoneq:
  .long 0x1e3a2220 // fcmp s17, s26
  b.ls .Lzeroq
  .long 0x1e311b74 // fdiv s20, s27, s17
  .long 0x4f949000 // fmul v0.4s, v0.4s, v20.s[0]
  .long 0x4f949021 // fmul v1.4s, v1.4s, v20.s[0]
  .long 0x4f949042 // fmul v2.4s, v2.4s, v20.s[0]
  .long 0x4f949063 // fmul v3.4s, v3.4s, v20.s[0]
  .long 0x4f949084 // fmul v4.4s, v4.4s, v20.s[0]
  .long 0x4f9490a5 // fmul v5.4s, v5.4s, v20.s[0]
  .long 0x4f9490c6 // fmul v6.4s, v6.4s, v20.s[0]
  .long 0x4f9490e7 // fmul v7.4s, v7.4s, v20.s[0]
  .long 0x4e21a800 // fcvtns v0.4s, v0.4s
  .long 0x4e21a821 // fcvtns v1.4s, v1.4s
  .long 0x4e21a842 // fcvtns v2.4s, v2.4s
  .long 0x4e21a863 // fcvtns v3.4s, v3.4s
  .long 0x4e21a884 // fcvtns v4.4s, v4.4s
  .long 0x4e21a8a5 // fcvtns v5.4s, v5.4s
  .long 0x4e21a8c6 // fcvtns v6.4s, v6.4s
  .long 0x4e21a8e7 // fcvtns v7.4s, v7.4s
  .long 0x0e614800 // sqxtn v0.4h, v0.4s
  .long 0x4e614820 // sqxtn2 v0.8h, v1.4s
  .long 0x0e614842 // sqxtn v2.4h, v2.4s
  .long 0x4e614862 // sqxtn2 v2.8h, v3.4s
  .long 0x0e614884 // sqxtn v4.4h, v4.4s
  .long 0x4e6148a4 // sqxtn2 v4.8h, v5.4s
  .long 0x0e6148c6 // sqxtn v6.4h, v6.4s
  .long 0x4e6148e6 // sqxtn2 v6.8h, v7.4s
  .long 0x0e214800 // sqxtn v0.8b, v0.8h
  .long 0x4e214840 // sqxtn2 v0.16b, v2.8h
  .long 0x0e214884 // sqxtn v4.8b, v4.8h
  .long 0x4e2148c4 // sqxtn2 v4.16b, v6.8h
  .long 0x3c802040 // stur q0, [x2, #2]
  .long 0x3c812044 // stur q4, [x2, #18]
  b .Lnextq
.Lzeroq:
  .long 0x4f00041c // movi v28.4s, #0x0
  .long 0x3c80205c // stur q28, [x2, #2]
  .long 0x3c81205c // stur q28, [x2, #18]
.Lnextq:
  add x2, x2, #34
  subs x1, x1, #1
  b.ne .Lloopq
.Lokq:
  mov x0, #0
  b .Lretq
.Lbadq:
  sub x0, x11, x1
  add x0, x0, #1
.Lretq:
end;

{$endif FPC}
{$endif CPUAARCH64}

procedure InitKernels;
var
  h: cardinal;
begin
  // as bits: the table holds NaN patterns too, which must not pass the x87 FPU
  for h := 0 to high(word) do
    PCardinal(@INFER_HALF[h])^ := InferHalfToSingleBits(h);
  InferQ8BlockDots := InferQ8BlockDotsPascal;
  InferQ8BlockDots4 := InferQ8BlockDots4Pascal;
  InferQ8KernelName := 'pascal';
  {$ifdef CPUAARCH64}
  {$ifdef FPC}
  // plain NEON (every aarch64 CPU): the float attention kernels
  InferDotsF32 := TInferDotsF32(@DotsF32Neon);
  InferAxpyRowsF32 := TInferDotsF32(@AxpyRowsF32Neon);
  InferQ8QuantizeRow := TInferQ8QuantizeRow(@Q8QuantizeRowNeon);
  if arm64ASIMDDP in CpuFeatures then
  begin
    // an assembler routine: one calling convention on aarch64 whatever the
    // procedure type says
    InferQ8BlockDots := TInferQ8BlockDots(@Q8BlockDotsNeonDot);
    InferQ8BlockDots4 := TInferQ8BlockDots4(@Q8BlockDots4NeonDot);
    InferQ8Linear4 := TInferQ8Linear4(@Q8Linear4NeonDot);
    InferQ8Linear4x2 := TInferQ8Linear4(@Q8Linear4x2NeonDot);
    InferQ8DotF32x4x2 := TInferQ8DotF32x4x2(@Q8DotF32x4x2NeonDot);
    InferQ8KernelName := 'neon-dotprod';
  end;
  {$endif FPC}
  {$endif CPUAARCH64}
end;

initialization
  InitKernels;

end.
