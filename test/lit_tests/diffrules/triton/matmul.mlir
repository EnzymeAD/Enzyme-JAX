// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=matmul_kernel outfn= argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const,enzyme_const retTys= mode=ForwardMode" --canonicalize | FileCheck %s

module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, 
                                %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, 
                                %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<32> : tensor<64x32xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
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
    %19 = arith.muli %12, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %21 = tt.splat %19 : i32 -> tensor<64xi32>
    %22 = arith.addi %21, %20 : tensor<64xi32>
    %23 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %24 = arith.remsi %22, %23 : tensor<64xi32>
    %25 = arith.muli %13, %c32_i32 : i32
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %27 = tt.splat %25 : i32 -> tensor<32xi32>
    %28 = arith.addi %27, %26 : tensor<32xi32>
    %29 = tt.splat %arg4 : i32 -> tensor<32xi32>
    %30 = arith.remsi %28, %29 : tensor<32xi32>
    %31 = tt.expand_dims %24 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %32 = tt.splat %arg6 : i32 -> tensor<64x1xi32>
    %33 = arith.muli %31, %32 : tensor<64x1xi32>
    %34 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %35 = tt.broadcast %33 : tensor<64x1xi32> -> tensor<64x32xi32>
    %36 = tt.broadcast %34 : tensor<1x32xi32> -> tensor<64x32xi32>
    %37 = arith.addi %35, %36 : tensor<64x32xi32>
    %38 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>
    %39 = tt.addptr %38, %37 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
    %40 = tt.expand_dims %26 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %41 = tt.splat %arg7 : i32 -> tensor<32x1xi32>
    %42 = arith.muli %40, %41 : tensor<32x1xi32>
    %43 = tt.expand_dims %30 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %44 = tt.broadcast %42 : tensor<32x1xi32> -> tensor<32x32xi32>
    %45 = tt.broadcast %43 : tensor<1x32xi32> -> tensor<32x32xi32>
    %46 = arith.addi %44, %45 : tensor<32x32xi32>
    %47 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %48 = tt.addptr %47, %46 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %49 = arith.addi %arg5, %c31_i32 : i32
    %50 = arith.divsi %49, %c32_i32 : i32
    %51:3 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_1, %arg11 = %39, %arg12 = %48) -> (tensor<64x32xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x32x!tt.ptr<f32>>)  : i32 {
      %69 = arith.muli %arg9, %c32_i32 : i32
      %70 = arith.subi %arg5, %69 : i32
      %71 = tt.splat %70 : i32 -> tensor<1x32xi32>
      %72 = arith.cmpi slt, %34, %71 : tensor<1x32xi32>
      %73 = tt.broadcast %72 : tensor<1x32xi1> -> tensor<64x32xi1>
      %74 = tt.load %arg11, %73, %cst_1 : tensor<64x32x!tt.ptr<f32>>
      %75 = tt.splat %70 : i32 -> tensor<32x1xi32>
      %76 = arith.cmpi slt, %40, %75 : tensor<32x1xi32>
      %77 = tt.broadcast %76 : tensor<32x1xi1> -> tensor<32x32xi1>
      %78 = tt.load %arg12, %77, %cst_0 : tensor<32x32x!tt.ptr<f32>>
      %79 = tt.dot %74, %78, %arg10, inputPrecision = tf32 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>
      %80 = tt.addptr %arg11, %cst : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
      %81 = arith.muli %arg7, %c32_i32 : i32
      %82 = tt.splat %81 : i32 -> tensor<32x32xi32>
      %83 = tt.addptr %arg12, %82 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
      scf.yield %79, %80, %83 : tensor<64x32xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x32x!tt.ptr<f32>>
    }
    %52 = arith.truncf %51#0 : tensor<64x32xf32> to tensor<64x32xf16>
    %53 = tt.expand_dims %22 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %54 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %55 = arith.muli %54, %53 : tensor<64x1xi32>
    %56 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
    %57 = tt.addptr %56, %55 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
    %58 = tt.expand_dims %28 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %59 = tt.broadcast %57 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x32x!tt.ptr<f16>>
    %60 = tt.broadcast %58 : tensor<1x32xi32> -> tensor<64x32xi32>
    %61 = tt.addptr %59, %60 : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
    %62 = tt.splat %arg3 : i32 -> tensor<64x1xi32>
    %63 = arith.cmpi slt, %53, %62 : tensor<64x1xi32>
    %64 = tt.splat %arg4 : i32 -> tensor<1x32xi32>
    %65 = arith.cmpi slt, %58, %64 : tensor<1x32xi32>
    %66 = tt.broadcast %63 : tensor<64x1xi1> -> tensor<64x32xi1>
    %67 = tt.broadcast %65 : tensor<1x32xi1> -> tensor<64x32xi1>
    %68 = arith.andi %66, %67 : tensor<64x32xi1>
    tt.store %61, %52, %68 : tensor<64x32x!tt.ptr<f16>>
    tt.return
  }
}
