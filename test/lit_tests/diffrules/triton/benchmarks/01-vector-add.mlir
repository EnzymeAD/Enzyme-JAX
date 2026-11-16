// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/01-vector-add.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=add_kernel outfn=fwd_add_kernel argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD-ADD-KERNEL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=add_kernel outfn=rev_add_kernel argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV-ADD-KERNEL

module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}


// FWD-ADD-KERNEL-LABEL: tt.func private @fwd_add_kernel(
// FWD-ADD-KERNEL: tt.store
// FWD-ADD-KERNEL: tt.return

// REV-ADD-KERNEL-LABEL: tt.func private @rev_add_kernel(
// REV-ADD-KERNEL: tt.atomic_rmw fadd, relaxed, gpu
// REV-ADD-KERNEL: tt.return
