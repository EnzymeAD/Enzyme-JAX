// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/06-fused-attention.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_attn_fwd outfn=fwd__attn_fwd argTys=enzyme_const,enzyme_dup,enzyme_const,enzyme_const,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD--ATTN-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_attn_fwd outfn=rev__attn_fwd argTys=enzyme_const,enzyme_dup,enzyme_const,enzyme_const,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV--ATTN-FWD

module {
  tt.func public @_attn_fwd(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32>
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %arg2, %arg3 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = tt.make_tensor_descriptor %arg4, [%5, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <64x64xf16>
    %7 = tt.make_tensor_descriptor %arg6, [%5, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <32x64xf16>
    %8 = tt.make_tensor_descriptor %arg5, [%5, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <32x64xf16>
    %9 = tt.make_tensor_descriptor %arg7, [%5, %c64_i32], [%c64_i64, %c1_i64] : <f16>, <64x64xf16>
    %10 = arith.muli %arg8, %arg3 : i32
    %11 = arith.muli %2, %10 : i32
    %12 = arith.muli %3, %arg8 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = arith.muli %0, %c64_i32 : i32
    %15 = arith.addi %13, %14 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %17 = tt.splat %14 : i32 -> tensor<64xi32>
    %18 = arith.addi %17, %16 : tensor<64xi32>
    %19 = arith.mulf %arg0, %cst_3 : f32
    %20 = tt.descriptor_load %6[%15, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
    %21:5 = scf.for %arg9 = %c0_i32 to %arg8 step %c32_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst, %arg12 = %cst_0, %arg13 = %13, %arg14 = %13) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32)  : i32 {
      %32 = tt.descriptor_load %8[%arg13, %c0_i32] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
      %33 = tt.trans %32 {order = array<i32: 1, 0>} : tensor<32x64xf16> -> tensor<64x32xf16>
      %34 = tt.dot %20, %33, %cst_1, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x32xf16> -> tensor<64x32xf32>
      %35 = "tt.reduce"(%34) <{axis = 1 : i32}> ({
      ^bb0(%arg15: f32, %arg16: f32):
        %58 = arith.maxnumf %arg15, %arg16 : f32
        tt.reduce.return %58 : f32
      }) : (tensor<64x32xf32>) -> tensor<64xf32>
      %36 = tt.splat %19 : f32 -> tensor<64xf32>
      %37 = arith.mulf %35, %36 : tensor<64xf32>
      %38 = arith.maxnumf %arg12, %37 : tensor<64xf32>
      %39 = tt.splat %19 : f32 -> tensor<64x32xf32>
      %40 = arith.mulf %34, %39 : tensor<64x32xf32>
      %41 = tt.expand_dims %38 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %42 = tt.broadcast %41 : tensor<64x1xf32> -> tensor<64x32xf32>
      %43 = arith.subf %40, %42 : tensor<64x32xf32>
      %44 = math.exp2 %43 : tensor<64x32xf32>
      %45 = arith.subf %arg12, %38 : tensor<64xf32>
      %46 = math.exp2 %45 : tensor<64xf32>
      %47 = "tt.reduce"(%44) <{axis = 1 : i32}> ({
      ^bb0(%arg15: f32, %arg16: f32):
        %58 = arith.addf %arg15, %arg16 : f32
        tt.reduce.return %58 : f32
      }) : (tensor<64x32xf32>) -> tensor<64xf32>
      %48 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %49 = tt.broadcast %48 : tensor<64x1xf32> -> tensor<64x64xf32>
      %50 = arith.mulf %arg10, %49 : tensor<64x64xf32>
      %51 = tt.descriptor_load %7[%arg14, %c0_i32] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
      %52 = arith.truncf %44 : tensor<64x32xf32> to tensor<64x32xf16>
      %53 = tt.dot %52, %51, %50, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
      %54 = arith.mulf %arg11, %46 : tensor<64xf32>
      %55 = arith.addf %54, %47 : tensor<64xf32>
      %56 = arith.addi %arg13, %c32_i32 : i32
      %57 = arith.addi %arg14, %c32_i32 : i32
      scf.yield %53, %55, %38, %56, %57 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32
    }
    %22 = math.log2 %21#1 : tensor<64xf32>
    %23 = arith.addf %21#2, %22 : tensor<64xf32>
    %24 = tt.expand_dims %21#1 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %25 = tt.broadcast %24 : tensor<64x1xf32> -> tensor<64x64xf32>
    %26 = arith.divf %21#0, %25 : tensor<64x64xf32>
    %27 = arith.muli %1, %arg8 : i32
    %28 = tt.addptr %arg1, %27 : !tt.ptr<f32>, i32
    %29 = tt.splat %28 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %30 = tt.addptr %29, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %30, %23 : tensor<64x!tt.ptr<f32>>
    %31 = arith.truncf %26 : tensor<64x64xf32> to tensor<64x64xf16>
    tt.descriptor_store %9[%15, %c0_i32], %31 : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
    tt.return
  }
}


// FWD--ATTN-FWD-LABEL: tt.func private @fwd__attn_fwd(
// FWD--ATTN-FWD: tt.dot
// FWD--ATTN-FWD: tt.dot
// FWD--ATTN-FWD: "tt.reduce"
// FWD--ATTN-FWD: tt.store
// FWD--ATTN-FWD: tt.return

// REV--ATTN-FWD-LABEL: tt.func private @rev__attn_fwd(
// REV--ATTN-FWD: tt.trans
// REV--ATTN-FWD: tt.dot
// REV--ATTN-FWD: tt.descriptor_reduce add
// REV--ATTN-FWD: tt.return
