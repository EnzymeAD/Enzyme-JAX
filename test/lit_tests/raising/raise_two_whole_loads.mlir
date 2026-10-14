// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false" --arith-raise --canonicalize | FileCheck %s

// Two loads of the whole of one buffer over different loop variables,
// b[k + i * 4] and b[k + j * 4], are the buffer's tensor read along (i, k)
// and along (j, k): their product is the outer product over (i, k, j),
// reduced over k. (The second load was recorded over the first's
// variables, and the product read as the square along (i, k).)
module {
  func.func private @two_loads(%b: memref<12xf64, 1>, %a: memref<9xf64, 1>) {
    affine.parallel (%i1) = (0) to (3) {
      affine.parallel (%j2) = (0) to (3) {
        %val = affine.parallel (%k1) = (0) to (4) reduce ("addf") -> (f64) {
          %bi1 = affine.load %b[%k1 + %i1 * 4] : memref<12xf64, 1>
          %bj2 = affine.load %b[%k1 + %j2 * 4] : memref<12xf64, 1>
          %p1 = arith.mulf %bi1, %bj2 : f64
          affine.yield %p1 : f64
        }
        affine.store %val, %a[%i1 + %j2 * 3] : memref<9xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func private @two_loads_raised(%arg0: tensor<12xf64>, %arg1: tensor<9xf64>) -> (tensor<12xf64>, tensor<9xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<12xf64>) -> tensor<12xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<12xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %0 : (tensor<12xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<3x4xf64>) -> tensor<3x4x3xf64>
// CHECK-NEXT:   %4 = stablehlo.broadcast_in_dim %2, dims = [2, 1] : (tensor<3x4xf64>) -> tensor<3x4x3xf64>
// CHECK-NEXT:   %5 = stablehlo.multiply %3, %4 : tensor<3x4x3xf64>
// CHECK-NEXT:   %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x4x3xf64>, tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:   %7 = stablehlo.broadcast_in_dim %6, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<3x3xf64>) -> tensor<9xf64>
// CHECK-NEXT:   %9 = stablehlo.dynamic_update_slice %arg1, %8, %c : (tensor<9xf64>, tensor<9xf64>, tensor<i64>) -> tensor<9xf64>
// CHECK-NEXT:   return %arg0, %9 : tensor<12xf64>, tensor<9xf64>
// CHECK-NEXT: }
