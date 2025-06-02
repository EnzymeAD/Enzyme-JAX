// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func private @kernel(%arg0: memref<26x1x1xf64, 1>, %arg1: memref<1x32x48xf64, 1>) {
    %0 = ub.poison : f64
    %1 = ub.poison : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %2:3 = affine.for %arg2 = 0 to 31 iter_args(%arg3 = %cst, %arg4 = %1, %arg5 = %0) -> (f64, i64, f64) {
      %8 = arith.index_cast %arg2 : index to i64
      %9 = arith.addi %8, %c1_i64 : i64
      %10 = affine.load %arg1[0, 8, %arg2 + 8] : memref<1x32x48xf64, 1>
      %11 = arith.addf %arg3, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.yield %11, %9, %11 : f64, i64, f64
    }
    %3 = arith.sitofp %2#1 : i64 to f64
    %4 = arith.divf %2#2, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
    %5 = arith.cmpi ne, %2#1, %c0_i64 : i64
    %6 = arith.cmpi eq, %2#1, %c0_i64 : i64
    %7 = scf.if %6 -> (f64) {
      %8 = arith.bitcast %4 : f64 to i64
      %9 = arith.select %5, %8, %c0_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
      %10 = arith.sitofp %9 : i64 to f64
      scf.yield %10 : f64
    } else {
      %8 = arith.select %5, %4, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      scf.yield %8 : f64
    }
    affine.store %7, %arg0[18, 0, 0] : memref<26x1x1xf64, 1>
    return
  }
}

// CHECK:  func.func private @kernel_raised(%arg0: tensor<26x1x1xf64>, %arg1: tensor<1x32x48xf64>) -> (tensor<26x1x1xf64>, tensor<1x32x48xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.100000e+01> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 8:9, 8:9] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg1 [0:1, 8:9, 9:10] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %4 = stablehlo.add %1, %3 : tensor<f64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg1 [0:1, 8:9, 10:11] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %7 = stablehlo.add %4, %6 : tensor<f64>
// CHECK-NEXT:    %8 = stablehlo.slice %arg1 [0:1, 8:9, 11:12] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %10 = stablehlo.add %7, %9 : tensor<f64>
// CHECK-NEXT:    %11 = stablehlo.slice %arg1 [0:1, 8:9, 12:13] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %13 = stablehlo.add %10, %12 : tensor<f64>
// CHECK-NEXT:    %14 = stablehlo.slice %arg1 [0:1, 8:9, 13:14] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %16 = stablehlo.add %13, %15 : tensor<f64>
// CHECK-NEXT:    %17 = stablehlo.slice %arg1 [0:1, 8:9, 14:15] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %18 = stablehlo.reshape %17 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %19 = stablehlo.add %16, %18 : tensor<f64>
// CHECK-NEXT:    %20 = stablehlo.slice %arg1 [0:1, 8:9, 15:16] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %21 = stablehlo.reshape %20 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %22 = stablehlo.add %19, %21 : tensor<f64>
// CHECK-NEXT:    %23 = stablehlo.slice %arg1 [0:1, 8:9, 16:17] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %24 = stablehlo.reshape %23 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %25 = stablehlo.add %22, %24 : tensor<f64>
// CHECK-NEXT:    %26 = stablehlo.slice %arg1 [0:1, 8:9, 17:18] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %27 = stablehlo.reshape %26 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %28 = stablehlo.add %25, %27 : tensor<f64>
// CHECK-NEXT:    %29 = stablehlo.slice %arg1 [0:1, 8:9, 18:19] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %30 = stablehlo.reshape %29 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %31 = stablehlo.add %28, %30 : tensor<f64>
// CHECK-NEXT:    %32 = stablehlo.slice %arg1 [0:1, 8:9, 19:20] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %33 = stablehlo.reshape %32 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %34 = stablehlo.add %31, %33 : tensor<f64>
// CHECK-NEXT:    %35 = stablehlo.slice %arg1 [0:1, 8:9, 20:21] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %36 = stablehlo.reshape %35 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %37 = stablehlo.add %34, %36 : tensor<f64>
// CHECK-NEXT:    %38 = stablehlo.slice %arg1 [0:1, 8:9, 21:22] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %39 = stablehlo.reshape %38 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %40 = stablehlo.add %37, %39 : tensor<f64>
// CHECK-NEXT:    %41 = stablehlo.slice %arg1 [0:1, 8:9, 22:23] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %42 = stablehlo.reshape %41 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %43 = stablehlo.add %40, %42 : tensor<f64>
// CHECK-NEXT:    %44 = stablehlo.slice %arg1 [0:1, 8:9, 23:24] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %45 = stablehlo.reshape %44 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %46 = stablehlo.add %43, %45 : tensor<f64>
// CHECK-NEXT:    %47 = stablehlo.slice %arg1 [0:1, 8:9, 24:25] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %48 = stablehlo.reshape %47 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %49 = stablehlo.add %46, %48 : tensor<f64>
// CHECK-NEXT:    %50 = stablehlo.slice %arg1 [0:1, 8:9, 25:26] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %51 = stablehlo.reshape %50 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %52 = stablehlo.add %49, %51 : tensor<f64>
// CHECK-NEXT:    %53 = stablehlo.slice %arg1 [0:1, 8:9, 26:27] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %54 = stablehlo.reshape %53 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %55 = stablehlo.add %52, %54 : tensor<f64>
// CHECK-NEXT:    %56 = stablehlo.slice %arg1 [0:1, 8:9, 27:28] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %57 = stablehlo.reshape %56 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %58 = stablehlo.add %55, %57 : tensor<f64>
// CHECK-NEXT:    %59 = stablehlo.slice %arg1 [0:1, 8:9, 28:29] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %60 = stablehlo.reshape %59 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %61 = stablehlo.add %58, %60 : tensor<f64>
// CHECK-NEXT:    %62 = stablehlo.slice %arg1 [0:1, 8:9, 29:30] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %63 = stablehlo.reshape %62 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %64 = stablehlo.add %61, %63 : tensor<f64>
// CHECK-NEXT:    %65 = stablehlo.slice %arg1 [0:1, 8:9, 30:31] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %66 = stablehlo.reshape %65 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %67 = stablehlo.add %64, %66 : tensor<f64>
// CHECK-NEXT:    %68 = stablehlo.slice %arg1 [0:1, 8:9, 31:32] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %69 = stablehlo.reshape %68 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %70 = stablehlo.add %67, %69 : tensor<f64>
// CHECK-NEXT:    %71 = stablehlo.slice %arg1 [0:1, 8:9, 32:33] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %72 = stablehlo.reshape %71 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %73 = stablehlo.add %70, %72 : tensor<f64>
// CHECK-NEXT:    %74 = stablehlo.slice %arg1 [0:1, 8:9, 33:34] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %75 = stablehlo.reshape %74 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %76 = stablehlo.add %73, %75 : tensor<f64>
// CHECK-NEXT:    %77 = stablehlo.slice %arg1 [0:1, 8:9, 34:35] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %78 = stablehlo.reshape %77 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %79 = stablehlo.add %76, %78 : tensor<f64>
// CHECK-NEXT:    %80 = stablehlo.slice %arg1 [0:1, 8:9, 35:36] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %81 = stablehlo.reshape %80 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %82 = stablehlo.add %79, %81 : tensor<f64>
// CHECK-NEXT:    %83 = stablehlo.slice %arg1 [0:1, 8:9, 36:37] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %84 = stablehlo.reshape %83 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %85 = stablehlo.add %82, %84 : tensor<f64>
// CHECK-NEXT:    %86 = stablehlo.slice %arg1 [0:1, 8:9, 37:38] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %87 = stablehlo.reshape %86 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %88 = stablehlo.add %85, %87 : tensor<f64>
// CHECK-NEXT:    %89 = stablehlo.slice %arg1 [0:1, 8:9, 38:39] : (tensor<1x32x48xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %90 = stablehlo.reshape %89 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %91 = stablehlo.add %88, %90 : tensor<f64>
// CHECK-NEXT:    %92 = stablehlo.divide %91, %cst : tensor<f64>
// CHECK-NEXT:    %93 = stablehlo.reshape %92 : (tensor<f64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %94 = stablehlo.slice %arg0 [0:18, 0:1, 0:1] : (tensor<26x1x1xf64>) -> tensor<18x1x1xf64>
// CHECK-NEXT:    %95 = stablehlo.slice %arg0 [19:26, 0:1, 0:1] : (tensor<26x1x1xf64>) -> tensor<7x1x1xf64>
// CHECK-NEXT:    %96 = stablehlo.concatenate %94, %93, %95, dim = 0 : (tensor<18x1x1xf64>, tensor<1x1x1xf64>, tensor<7x1x1xf64>) -> tensor<26x1x1xf64>
// CHECK-NEXT:    return %96, %arg1 : tensor<26x1x1xf64>, tensor<1x32x48xf64>
// CHECK-NEXT:  }
