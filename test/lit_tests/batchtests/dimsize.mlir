// RUN: enzymexlamlir-opt %s --enzyme-batch --inline | FileCheck %s

module {
  func.func private @unbatched_f2(%arg0: tensor<5x2xi64>) -> tensor<i64> {
    %1 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<5x2xi64>) -> tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
    return %2 : tensor<i64>
  }
  func.func @main(%arg0: tensor<3x5x2xi64>) -> (tensor<3xi64>) {
    %2 = enzyme.batch @unbatched_f2(%arg0) {batch_shape = array<i64: 3>} : (tensor<3x5x2xi64>) -> tensor<3xi64>
    return %2 : tensor<3xi64>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x5x2xi64>) -> tensor<3xi64> {
// CHECK-NEXT:     %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x5x2xi64>) -> tensor<i32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     %2 = stablehlo.convert %1 : (tensor<3xi32>) -> tensor<3xi64>
// CHECK-NEXT:     return %2 : tensor<3xi64>
// CHECK-NEXT: }
