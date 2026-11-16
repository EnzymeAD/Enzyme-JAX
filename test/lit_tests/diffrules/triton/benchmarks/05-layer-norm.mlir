// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/05-layer-norm.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_layer_norm_fwd_fused outfn=fwd__layer_norm_fwd_fused argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD--LAYER-NORM-FWD-FUSED
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_layer_norm_fwd_fused outfn=rev__layer_norm_fwd_fused argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV--LAYER-NORM-FWD-FUSED

module {
  tt.func public @_layer_norm_fwd_fused(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: f32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = scf.for %arg9 = %c0_i32 to %arg7 step %c64_i32 iter_args(%arg10 = %cst_0) -> (tensor<64xf32>)  : i32 {
      %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %17 = tt.splat %arg9 : i32 -> tensor<64xi32>
      %18 = arith.addi %17, %16 : tensor<64xi32>
      %19 = tt.splat %arg7 : i32 -> tensor<64xi32>
      %20 = arith.cmpi slt, %18, %19 : tensor<64xi32>
      %21 = tt.splat %3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %22 = tt.addptr %21, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %23 = tt.load %22, %20, %cst_0 : tensor<64x!tt.ptr<f32>>
      %24 = arith.addf %arg10, %23 : tensor<64xf32>
      scf.yield %24 : tensor<64xf32>
    }
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %16 : f32
    }) : (tensor<64xf32>) -> f32
    %6 = arith.sitofp %arg7 : i32 to f32
    %7 = arith.divf %5, %6 : f32
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c64_i32 iter_args(%arg10 = %cst_0) -> (tensor<64xf32>)  : i32 {
      %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %17 = tt.splat %arg9 : i32 -> tensor<64xi32>
      %18 = arith.addi %17, %16 : tensor<64xi32>
      %19 = tt.splat %arg7 : i32 -> tensor<64xi32>
      %20 = arith.cmpi slt, %18, %19 : tensor<64xi32>
      %21 = tt.splat %3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %22 = tt.addptr %21, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %23 = tt.load %22, %20, %cst_0 : tensor<64x!tt.ptr<f32>>
      %24 = tt.splat %7 : f32 -> tensor<64xf32>
      %25 = arith.subf %23, %24 : tensor<64xf32>
      %26 = arith.select %20, %25, %cst_0 : tensor<64xi1>, tensor<64xf32>
      %27 = arith.mulf %26, %26 : tensor<64xf32>
      %28 = arith.addf %arg10, %27 : tensor<64xf32>
      scf.yield %28 : tensor<64xf32>
    }
    %9 = "tt.reduce"(%8) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %16 : f32
    }) : (tensor<64xf32>) -> f32
    %10 = arith.divf %9, %6 : f32
    %11 = arith.addf %10, %arg8 : f32
    %12 = math.sqrt %11 : f32
    %13 = arith.divf %cst, %12 : f32
    %14 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %14, %7 : !tt.ptr<f32>
    %15 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %15, %13 : !tt.ptr<f32>
    scf.for %arg9 = %c0_i32 to %arg7 step %c64_i32  : i32 {
      %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %17 = tt.splat %arg9 : i32 -> tensor<64xi32>
      %18 = arith.addi %17, %16 : tensor<64xi32>
      %19 = tt.splat %arg7 : i32 -> tensor<64xi32>
      %20 = arith.cmpi slt, %18, %19 : tensor<64xi32>
      %21 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %22 = tt.addptr %21, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %23 = tt.load %22, %20 : tensor<64x!tt.ptr<f32>>
      %24 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %25 = tt.addptr %24, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %26 = tt.load %25, %20 : tensor<64x!tt.ptr<f32>>
      %27 = tt.splat %3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %28 = tt.addptr %27, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %29 = tt.load %28, %20, %cst_0 : tensor<64x!tt.ptr<f32>>
      %30 = tt.splat %7 : f32 -> tensor<64xf32>
      %31 = arith.subf %29, %30 : tensor<64xf32>
      %32 = tt.splat %13 : f32 -> tensor<64xf32>
      %33 = arith.mulf %31, %32 : tensor<64xf32>
      %34 = arith.mulf %33, %23 : tensor<64xf32>
      %35 = arith.addf %34, %26 : tensor<64xf32>
      %36 = tt.splat %2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %37 = tt.addptr %36, %18 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %37, %35, %20 : tensor<64x!tt.ptr<f32>>
    }
    tt.return
  }
}


// FWD--LAYER-NORM-FWD-FUSED-LABEL: tt.func private @fwd__layer_norm_fwd_fused(
// FWD--LAYER-NORM-FWD-FUSED: "tt.reduce"
// FWD--LAYER-NORM-FWD-FUSED: tt.store
// FWD--LAYER-NORM-FWD-FUSED: tt.return

// REV--LAYER-NORM-FWD-FUSED-LABEL: tt.func private @rev__layer_norm_fwd_fused(
// REV--LAYER-NORM-FWD-FUSED: tt.atomic_rmw fadd, relaxed, gpu
// REV--LAYER-NORM-FWD-FUSED: tt.return
