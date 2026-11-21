// RUN: enzymexlamlir-opt --enzyme-hlo-element-chunk="elements=5 chunk-bytes=64 barriers=true" %s | FileCheck %s

// Five elements of 32 bytes in chunks of at most 64 bytes: five is prime, so
// the unrolled chunks take 1, 2 and 2 elements rather than one each.
func.func @apply(%x: tensor<5x4xf64>, %b: tensor<4xf64>) -> tensor<5x4xf64> {
  %0 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf64>) -> tensor<5x4xf64>
  %1 = stablehlo.multiply %x, %0 : tensor<5x4xf64>
  return %1 : tensor<5x4xf64>
}

// CHECK-LABEL: func.func @apply(
// CHECK-SAME: %arg0: tensor<5x4xf64>, %arg1: tensor<4xf64>) -> tensor<5x4xf64> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x4xf64>
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [0:1, 0:4] : (tensor<5x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %2 = stablehlo.multiply %0, %1 : tensor<1x4xf64>
// CHECK-NEXT: %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %3 = stablehlo.dynamic_update_slice %cst, %2, %c, %c_0 : (tensor<5x4xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
// CHECK-NEXT: %4:3 = stablehlo.optimization_barrier %3, %arg1, %arg0 : tensor<5x4xf64>, tensor<4xf64>, tensor<5x4xf64>
// CHECK-NEXT: %5 = stablehlo.slice %4#2 [1:3, 0:4] : (tensor<5x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %6 = stablehlo.broadcast_in_dim %4#1, dims = [1] : (tensor<4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %7 = stablehlo.multiply %5, %6 : tensor<2x4xf64>
// CHECK-NEXT: %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %8 = stablehlo.dynamic_update_slice %4#0, %7, %c_1, %c_2 : (tensor<5x4xf64>, tensor<2x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
// CHECK-NEXT: %9:3 = stablehlo.optimization_barrier %8, %4#1, %4#2 : tensor<5x4xf64>, tensor<4xf64>, tensor<5x4xf64>
// CHECK-NEXT: %10 = stablehlo.slice %9#2 [3:5, 0:4] : (tensor<5x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %11 = stablehlo.broadcast_in_dim %9#1, dims = [1] : (tensor<4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT: %12 = stablehlo.multiply %10, %11 : tensor<2x4xf64>
// CHECK-NEXT: %c_3 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT: %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %13 = stablehlo.dynamic_update_slice %9#0, %12, %c_3, %c_4 : (tensor<5x4xf64>, tensor<2x4xf64>, tensor<i64>, tensor<i64>) -> tensor<5x4xf64>
// CHECK-NEXT: return %13 : tensor<5x4xf64>
// CHECK-NEXT: }
