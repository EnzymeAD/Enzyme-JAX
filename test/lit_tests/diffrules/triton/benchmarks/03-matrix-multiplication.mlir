// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/03-matrix-multiplication.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=matmul_kernel outfn=fwd_matmul_kernel argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD-MATMUL-KERNEL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=matmul_kernel outfn=rev_matmul_kernel argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV-MATMUL-KERNEL

module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<32> : tensor<32x32xi32>
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.addi %arg4, %c31_i32 : i32
    %4 = arith.divsi %3, %c32_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.cmpi sge, %12, %c0_i32 : i32
    llvm.intr.assume %14 : i1
    %15 = arith.cmpi sge, %13, %c0_i32 : i32
    llvm.intr.assume %15 : i1
    %16 = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %16 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %17 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    llvm.intr.assume %17 : i1
    %18 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %18 : i1
    llvm.intr.assume %true : i1
    %19 = arith.muli %12, %c32_i32 : i32
    %20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %21 = tt.splat %19 : i32 -> tensor<32xi32>
    %22 = arith.addi %21, %20 : tensor<32xi32>
    %23 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %24 = arith.remsi %22, %23 : tensor<32xi32>
    %25 = arith.muli %13, %c32_i32 : i32
    %26 = tt.splat %25 : i32 -> tensor<32xi32>
    %27 = arith.addi %26, %20 : tensor<32xi32>
    %28 = tt.splat %arg4 : i32 -> tensor<32xi32>
    %29 = arith.remsi %27, %28 : tensor<32xi32>
    %30 = tt.expand_dims %24 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %31 = tt.splat %arg6 : i32 -> tensor<32x1xi32>
    %32 = arith.muli %30, %31 : tensor<32x1xi32>
    %33 = tt.expand_dims %20 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %34 = tt.broadcast %32 : tensor<32x1xi32> -> tensor<32x32xi32>
    %35 = tt.broadcast %33 : tensor<1x32xi32> -> tensor<32x32xi32>
    %36 = arith.addi %34, %35 : tensor<32x32xi32>
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
    %38 = tt.addptr %37, %36 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
    %39 = tt.expand_dims %20 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %40 = tt.splat %arg7 : i32 -> tensor<32x1xi32>
    %41 = arith.muli %39, %40 : tensor<32x1xi32>
    %42 = tt.expand_dims %29 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %43 = tt.broadcast %41 : tensor<32x1xi32> -> tensor<32x32xi32>
    %44 = tt.broadcast %42 : tensor<1x32xi32> -> tensor<32x32xi32>
    %45 = arith.addi %43, %44 : tensor<32x32xi32>
    %46 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
    %47 = tt.addptr %46, %45 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
    %48 = arith.addi %arg5, %c31_i32 : i32
    %49 = arith.divsi %48, %c32_i32 : i32
    %50:3 = scf.for %arg9 = %c0_i32 to %49 step %c1_i32 iter_args(%arg10 = %38, %arg11 = %47, %arg12 = %cst) -> (tensor<32x32x!tt.ptr<f16>>, tensor<32x32x!tt.ptr<f16>>, tensor<32x32xf32>)  : i32 {
      %68 = arith.muli %arg9, %c32_i32 : i32
      %69 = arith.subi %arg5, %68 : i32
      %70 = tt.splat %69 : i32 -> tensor<1x32xi32>
      %71 = arith.cmpi slt, %33, %70 : tensor<1x32xi32>
      %72 = tt.broadcast %71 : tensor<1x32xi1> -> tensor<32x32xi1>
      %73 = tt.load %arg10, %72, %cst_0 : tensor<32x32x!tt.ptr<f16>>
      %74 = tt.splat %69 : i32 -> tensor<32x1xi32>
      %75 = arith.cmpi slt, %39, %74 : tensor<32x1xi32>
      %76 = tt.broadcast %75 : tensor<32x1xi1> -> tensor<32x32xi1>
      %77 = tt.load %arg11, %76, %cst_0 : tensor<32x32x!tt.ptr<f16>>
      %78 = tt.dot %73, %77, %arg12, inputPrecision = tf32 : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      %79 = tt.addptr %arg10, %cst_1 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %80 = arith.muli %arg7, %c32_i32 : i32
      %81 = tt.splat %80 : i32 -> tensor<32x32xi32>
      %82 = tt.addptr %arg11, %81 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      scf.yield %79, %82, %78 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32x!tt.ptr<f16>>, tensor<32x32xf32>
    }
    %51 = arith.truncf %50#2 : tensor<32x32xf32> to tensor<32x32xf16>
    %52 = tt.expand_dims %22 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %53 = tt.splat %arg8 : i32 -> tensor<32x1xi32>
    %54 = arith.muli %53, %52 : tensor<32x1xi32>
    %55 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
    %56 = tt.addptr %55, %54 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
    %57 = tt.expand_dims %27 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %58 = tt.broadcast %56 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x32x!tt.ptr<f16>>
    %59 = tt.broadcast %57 : tensor<1x32xi32> -> tensor<32x32xi32>
    %60 = tt.addptr %58, %59 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
    %61 = tt.splat %arg3 : i32 -> tensor<32x1xi32>
    %62 = arith.cmpi slt, %52, %61 : tensor<32x1xi32>
    %63 = tt.splat %arg4 : i32 -> tensor<1x32xi32>
    %64 = arith.cmpi slt, %57, %63 : tensor<1x32xi32>
    %65 = tt.broadcast %62 : tensor<32x1xi1> -> tensor<32x32xi1>
    %66 = tt.broadcast %64 : tensor<1x32xi1> -> tensor<32x32xi1>
    %67 = arith.andi %65, %66 : tensor<32x32xi1>
    tt.store %60, %51, %67 : tensor<32x32x!tt.ptr<f16>>
    tt.return
  }
}


// FWD-MATMUL-KERNEL-LABEL: tt.func private @fwd_matmul_kernel(
// FWD-MATMUL-KERNEL: tt.dot
// FWD-MATMUL-KERNEL: tt.dot
// FWD-MATMUL-KERNEL: tt.store
// FWD-MATMUL-KERNEL: tt.return

// REV-MATMUL-KERNEL-LABEL: tt.func private @rev_matmul_kernel(
// REV-MATMUL-KERNEL: tt.trans
// REV-MATMUL-KERNEL: tt.dot
// REV-MATMUL-KERNEL: tt.atomic_rmw fadd, relaxed, gpu
// REV-MATMUL-KERNEL: tt.return
