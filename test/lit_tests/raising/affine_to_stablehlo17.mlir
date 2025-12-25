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
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.032258064516129031> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 8:9, 8:39] : (tensor<1x32x48xf64>) -> tensor<1x1x31xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x31xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %cst : tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<f64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [0:18, 0:1, 0:1] : (tensor<26x1x1xf64>) -> tensor<18x1x1xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg0 [19:26, 0:1, 0:1] : (tensor<26x1x1xf64>) -> tensor<7x1x1xf64>
// CHECK-NEXT:    %6 = stablehlo.concatenate %4, %3, %5, dim = 0 : (tensor<18x1x1xf64>, tensor<1x1x1xf64>, tensor<7x1x1xf64>) -> tensor<26x1x1xf64>
// CHECK-NEXT:    return %6, %arg1 : tensor<26x1x1xf64>, tensor<1x32x48xf64>
// CHECK-NEXT:  }
