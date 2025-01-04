// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
    %0 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
    %1 = arith.addf %0, %arg1 : tensor<2xf64>
    return %1 : tensor<2xf64>
  }
}

// CHECK:   func.func @main(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[i0:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     %[[i1:.+]] = stablehlo.add %[[i0:.+]], %arg1 : tensor<2xf64>
// CHECK-NEXT:     return %[[i1:.+]] : tensor<2xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
