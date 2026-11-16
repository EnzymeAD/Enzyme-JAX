// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Three updates into one buffer, each read back through a slice that
// overlaps the next update: the provenance walk behind DUSDUSSubsuming
// follows the whole chain.
func.func @chain(%arg0: tensor<16xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<16xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<2> : tensor<i64>
  %d0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c0 : (tensor<16xf64>, tensor<4xf64>, tensor<i64>) -> tensor<16xf64>
  %s0 = stablehlo.slice %d0 [3:7] : (tensor<16xf64>) -> tensor<4xf64>
  %c1 = stablehlo.constant dense<4> : tensor<i64>
  %d1 = stablehlo.dynamic_update_slice %d0, %arg2, %c1 : (tensor<16xf64>, tensor<4xf64>, tensor<i64>) -> tensor<16xf64>
  %s1 = stablehlo.slice %d1 [5:9] : (tensor<16xf64>) -> tensor<4xf64>
  %a1 = stablehlo.add %s0, %s1 : tensor<4xf64>
  %c2 = stablehlo.constant dense<6> : tensor<i64>
  %d2 = stablehlo.dynamic_update_slice %d1, %arg3, %c2 : (tensor<16xf64>, tensor<4xf64>, tensor<i64>) -> tensor<16xf64>
  %s2 = stablehlo.slice %d2 [7:11] : (tensor<16xf64>) -> tensor<4xf64>
  %a2 = stablehlo.add %a1, %s2 : tensor<4xf64>
  return %d2, %a2 : tensor<16xf64>, tensor<4xf64>
}

// CHECK:    func.func @chain(%arg0: tensor<16xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<16xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2] : (tensor<16xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [1:4] : (tensor<4xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [6:7] : (tensor<16xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %arg1 [0:2] : (tensor<4xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg2 [1:4] : (tensor<4xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [8:9] : (tensor<16xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %7 = stablehlo.concatenate %5, %6, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %8 = stablehlo.add %3, %7 : tensor<4xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %arg2 [0:2] : (tensor<4xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %10 = stablehlo.slice %arg0 [10:16] : (tensor<16xf64>) -> tensor<6xf64>
// CHECK-NEXT:    %11 = stablehlo.concatenate %0, %4, %9, %arg3, %10, dim = 0 : (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<4xf64>, tensor<6xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %12 = stablehlo.slice %arg3 [1:4] : (tensor<4xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %13 = stablehlo.slice %arg0 [10:11] : (tensor<16xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %14 = stablehlo.concatenate %12, %13, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %15 = stablehlo.add %8, %14 : tensor<4xf64>
// CHECK-NEXT:    return %11, %15 : tensor<16xf64>, tensor<4xf64>
// CHECK-NEXT:  }
