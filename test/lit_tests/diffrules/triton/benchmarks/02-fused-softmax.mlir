// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/02-fused-softmax.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=softmax_kernel outfn=fwd_softmax_kernel argTys=enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD-SOFTMAX-KERNEL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=softmax_kernel outfn=rev_softmax_kernel argTys=enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV-SOFTMAX-KERNEL

module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    scf.for %arg6 = %0 to %arg4 step %1  : i32 {
      %2 = arith.muli %arg6, %arg2 : i32
      %3 = tt.addptr %arg1, %2 : !tt.ptr<f32>, i32
      %4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %5 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %7 = tt.splat %arg5 : i32 -> tensor<1024xi32>
      %8 = arith.cmpi slt, %4, %7 : tensor<1024xi32>
      %9 = tt.load %6, %8, %cst : tensor<1024x!tt.ptr<f32>>
      %10 = "tt.reduce"(%9) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %21 = arith.maxnumf %arg7, %arg8 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<1024xf32>) -> f32
      %11 = tt.splat %10 : f32 -> tensor<1024xf32>
      %12 = arith.subf %9, %11 : tensor<1024xf32>
      %13 = math.exp %12 : tensor<1024xf32>
      %14 = "tt.reduce"(%13) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %21 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<1024xf32>) -> f32
      %15 = tt.splat %14 : f32 -> tensor<1024xf32>
      %16 = arith.divf %13, %15 : tensor<1024xf32>
      %17 = arith.muli %arg6, %arg3 : i32
      %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
      %19 = tt.splat %18 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %20 = tt.addptr %19, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %20, %16, %8 : tensor<1024x!tt.ptr<f32>>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}


// FWD-SOFTMAX-KERNEL-LABEL: tt.func private @fwd_softmax_kernel(
// FWD-SOFTMAX-KERNEL: "tt.reduce"
// FWD-SOFTMAX-KERNEL: tt.store
// FWD-SOFTMAX-KERNEL: tt.return

// REV-SOFTMAX-KERNEL-LABEL: tt.func private @rev_softmax_kernel(
// REV-SOFTMAX-KERNEL: tt.atomic_rmw fadd, relaxed, gpu
// REV-SOFTMAX-KERNEL: tt.return
