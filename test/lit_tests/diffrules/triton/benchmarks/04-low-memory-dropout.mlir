// REQUIRES: triton
// TTIR snapshot from triton/benchmark/mlir/04-low-memory-dropout.mlir.
// Differentiate the original kernel, retaining its masks, loops and descriptors.
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_dropout outfn=fwd__dropout argTys=enzyme_dup,enzyme_const,enzyme_dup,enzyme_const,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD--DROPOUT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_dropout outfn=rev__dropout argTys=enzyme_dup,enzyme_const,enzyme_dup,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV--DROPOUT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_seeded_dropout outfn=fwd__seeded_dropout argTys=enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD--SEEDED-DROPOUT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=_seeded_dropout outfn=rev__seeded_dropout argTys=enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV--SEEDED-DROPOUT

module {
  tt.func public @_dropout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: f32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_0 = arith.constant dense<0> : tensor<256xi32>
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %12 = tt.load %11, %6 : tensor<256x!tt.ptr<i32>>
    %13 = arith.subf %cst_1, %arg4 : f32
    %14 = tt.splat %13 : f32 -> tensor<256xf32>
    %15 = arith.divf %9, %14 : tensor<256xf32>
    %16 = arith.cmpi ne, %12, %cst_0 : tensor<256xi32>
    %17 = arith.select %16, %15, %cst : tensor<256xi1>, tensor<256xf32>
    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %19 = tt.addptr %18, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %19, %17, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
  tt.func public @_seeded_dropout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: f32, %arg4: i32) attributes {noinline = false} {
    %c-766435501_i32 = arith.constant -766435501 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-845247145_i32 = arith.constant -845247145 : i32
    %c32_i64 = arith.constant 32 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %c-1879881855_i32 = arith.constant -1879881855 : i32
    %c-616729560_i32 = arith.constant -616729560 : i32
    %c534103459_i32 = arith.constant 534103459 : i32
    %c1401181199_i32 = arith.constant 1401181199 : i32
    %c1684936478_i32 = arith.constant 1684936478 : i32
    %c-1253254570_i32 = arith.constant -1253254570 : i32
    %c-1459197799_i32 = arith.constant -1459197799 : i32
    %c387276957_i32 = arith.constant 387276957 : i32
    %c-308364780_i32 = arith.constant -308364780 : i32
    %c2027808484_i32 = arith.constant 2027808484 : i32
    %c842468239_i32 = arith.constant 842468239 : i32
    %c-626627285_i32 = arith.constant -626627285 : i32
    %c1993301258_i32 = arith.constant 1993301258 : i32
    %c1013904242_i32 = arith.constant 1013904242 : i32
    %c-1150833019_i32 = arith.constant -1150833019 : i32
    %c-1640531527_i32 = arith.constant -1640531527 : i32
    %cst = arith.constant dense<-766435501> : tensor<256xi32>
    %cst_0 = arith.constant dense<-845247145> : tensor<256xi32>
    %cst_1 = arith.constant dense<0> : tensor<256xi32>
    %cst_2 = arith.constant dense<4.6566126E-10> : tensor<256xf32>
    %cst_3 = arith.constant dense<-1> : tensor<256xi32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_5 = arith.constant 1.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.shrui %10, %c32_i64 : i64
    %12 = arith.andi %11, %c4294967295_i64 : i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.andi %10, %c4294967295_i64 : i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = tt.mulhiui %c-845247145_i32, %c0_i32 : i32
    %17 = arith.xori %16, %15 : i32
    %18 = tt.mulhiui %cst, %4 : tensor<256xi32>
    %19 = tt.splat %13 : i32 -> tensor<256xi32>
    %20 = arith.xori %18, %19 : tensor<256xi32>
    %21 = arith.muli %4, %cst : tensor<256xi32>
    %22 = arith.addi %15, %c-1640531527_i32 : i32
    %23 = arith.addi %13, %c-1150833019_i32 : i32
    %24 = tt.mulhiui %cst_0, %20 : tensor<256xi32>
    %25 = tt.splat %22 : i32 -> tensor<256xi32>
    %26 = arith.xori %24, %25 : tensor<256xi32>
    %27 = tt.mulhiui %c-766435501_i32, %17 : i32
    %28 = tt.splat %27 : i32 -> tensor<256xi32>
    %29 = arith.xori %28, %21 : tensor<256xi32>
    %30 = tt.splat %23 : i32 -> tensor<256xi32>
    %31 = arith.xori %29, %30 : tensor<256xi32>
    %32 = arith.muli %20, %cst_0 : tensor<256xi32>
    %33 = arith.muli %17, %c-766435501_i32 : i32
    %34 = tt.splat %33 : i32 -> tensor<256xi32>
    %35 = arith.addi %15, %c1013904242_i32 : i32
    %36 = arith.addi %13, %c1993301258_i32 : i32
    %37 = tt.mulhiui %cst_0, %31 : tensor<256xi32>
    %38 = arith.xori %37, %32 : tensor<256xi32>
    %39 = tt.splat %35 : i32 -> tensor<256xi32>
    %40 = arith.xori %38, %39 : tensor<256xi32>
    %41 = tt.mulhiui %cst, %26 : tensor<256xi32>
    %42 = arith.xori %41, %34 : tensor<256xi32>
    %43 = tt.splat %36 : i32 -> tensor<256xi32>
    %44 = arith.xori %42, %43 : tensor<256xi32>
    %45 = arith.muli %31, %cst_0 : tensor<256xi32>
    %46 = arith.muli %26, %cst : tensor<256xi32>
    %47 = arith.addi %15, %c-626627285_i32 : i32
    %48 = arith.addi %13, %c842468239_i32 : i32
    %49 = tt.mulhiui %cst_0, %44 : tensor<256xi32>
    %50 = arith.xori %49, %45 : tensor<256xi32>
    %51 = tt.splat %47 : i32 -> tensor<256xi32>
    %52 = arith.xori %50, %51 : tensor<256xi32>
    %53 = tt.mulhiui %cst, %40 : tensor<256xi32>
    %54 = arith.xori %53, %46 : tensor<256xi32>
    %55 = tt.splat %48 : i32 -> tensor<256xi32>
    %56 = arith.xori %54, %55 : tensor<256xi32>
    %57 = arith.muli %44, %cst_0 : tensor<256xi32>
    %58 = arith.muli %40, %cst : tensor<256xi32>
    %59 = arith.addi %15, %c2027808484_i32 : i32
    %60 = arith.addi %13, %c-308364780_i32 : i32
    %61 = tt.mulhiui %cst_0, %56 : tensor<256xi32>
    %62 = arith.xori %61, %57 : tensor<256xi32>
    %63 = tt.splat %59 : i32 -> tensor<256xi32>
    %64 = arith.xori %62, %63 : tensor<256xi32>
    %65 = tt.mulhiui %cst, %52 : tensor<256xi32>
    %66 = arith.xori %65, %58 : tensor<256xi32>
    %67 = tt.splat %60 : i32 -> tensor<256xi32>
    %68 = arith.xori %66, %67 : tensor<256xi32>
    %69 = arith.muli %56, %cst_0 : tensor<256xi32>
    %70 = arith.muli %52, %cst : tensor<256xi32>
    %71 = arith.addi %15, %c387276957_i32 : i32
    %72 = arith.addi %13, %c-1459197799_i32 : i32
    %73 = tt.mulhiui %cst_0, %68 : tensor<256xi32>
    %74 = arith.xori %73, %69 : tensor<256xi32>
    %75 = tt.splat %71 : i32 -> tensor<256xi32>
    %76 = arith.xori %74, %75 : tensor<256xi32>
    %77 = tt.mulhiui %cst, %64 : tensor<256xi32>
    %78 = arith.xori %77, %70 : tensor<256xi32>
    %79 = tt.splat %72 : i32 -> tensor<256xi32>
    %80 = arith.xori %78, %79 : tensor<256xi32>
    %81 = arith.muli %68, %cst_0 : tensor<256xi32>
    %82 = arith.muli %64, %cst : tensor<256xi32>
    %83 = arith.addi %15, %c-1253254570_i32 : i32
    %84 = arith.addi %13, %c1684936478_i32 : i32
    %85 = tt.mulhiui %cst_0, %80 : tensor<256xi32>
    %86 = arith.xori %85, %81 : tensor<256xi32>
    %87 = tt.splat %83 : i32 -> tensor<256xi32>
    %88 = arith.xori %86, %87 : tensor<256xi32>
    %89 = tt.mulhiui %cst, %76 : tensor<256xi32>
    %90 = arith.xori %89, %82 : tensor<256xi32>
    %91 = tt.splat %84 : i32 -> tensor<256xi32>
    %92 = arith.xori %90, %91 : tensor<256xi32>
    %93 = arith.muli %80, %cst_0 : tensor<256xi32>
    %94 = arith.muli %76, %cst : tensor<256xi32>
    %95 = arith.addi %15, %c1401181199_i32 : i32
    %96 = arith.addi %13, %c534103459_i32 : i32
    %97 = tt.mulhiui %cst_0, %92 : tensor<256xi32>
    %98 = arith.xori %97, %93 : tensor<256xi32>
    %99 = tt.splat %95 : i32 -> tensor<256xi32>
    %100 = arith.xori %98, %99 : tensor<256xi32>
    %101 = tt.mulhiui %cst, %88 : tensor<256xi32>
    %102 = arith.xori %101, %94 : tensor<256xi32>
    %103 = tt.splat %96 : i32 -> tensor<256xi32>
    %104 = arith.xori %102, %103 : tensor<256xi32>
    %105 = arith.muli %88, %cst : tensor<256xi32>
    %106 = arith.addi %13, %c-616729560_i32 : i32
    %107 = tt.mulhiui %cst, %100 : tensor<256xi32>
    %108 = arith.xori %107, %105 : tensor<256xi32>
    %109 = tt.splat %106 : i32 -> tensor<256xi32>
    %110 = arith.xori %108, %109 : tensor<256xi32>
    %111 = arith.muli %104, %cst_0 : tensor<256xi32>
    %112 = arith.addi %15, %c-1879881855_i32 : i32
    %113 = tt.mulhiui %cst_0, %110 : tensor<256xi32>
    %114 = arith.xori %113, %111 : tensor<256xi32>
    %115 = tt.splat %112 : i32 -> tensor<256xi32>
    %116 = arith.xori %114, %115 : tensor<256xi32>
    %117 = arith.cmpi slt, %116, %cst_1 : tensor<256xi32>
    %118 = arith.xori %116, %cst_3 : tensor<256xi32>
    %119 = arith.select %117, %118, %116 : tensor<256xi1>, tensor<256xi32>
    %120 = arith.sitofp %119 : tensor<256xi32> to tensor<256xf32>
    %121 = arith.mulf %120, %cst_2 : tensor<256xf32>
    %122 = tt.splat %arg3 : f32 -> tensor<256xf32>
    %123 = arith.cmpf ogt, %121, %122 : tensor<256xf32>
    %124 = arith.subf %cst_5, %arg3 : f32
    %125 = tt.splat %124 : f32 -> tensor<256xf32>
    %126 = arith.divf %9, %125 : tensor<256xf32>
    %127 = arith.select %123, %126, %cst_4 : tensor<256xi1>, tensor<256xf32>
    %128 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %129 = tt.addptr %128, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %129, %127, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}


// FWD--DROPOUT-LABEL: tt.func private @fwd__dropout(
// FWD--DROPOUT: tt.store
// FWD--DROPOUT: tt.return

// REV--DROPOUT-LABEL: tt.func private @rev__dropout(
// REV--DROPOUT: tt.atomic_rmw fadd, relaxed, gpu
// REV--DROPOUT: tt.return

// FWD--SEEDED-DROPOUT-LABEL: tt.func private @fwd__seeded_dropout(
// FWD--SEEDED-DROPOUT: tt.store
// FWD--SEEDED-DROPOUT: tt.return

// REV--SEEDED-DROPOUT-LABEL: tt.func private @rev__seeded_dropout(
// REV--SEEDED-DROPOUT: tt.atomic_rmw fadd, relaxed, gpu
// REV--SEEDED-DROPOUT: tt.return
