// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<20x85x180xf64>) -> tensor<85x180x20xf64> {
    %456 = "stablehlo.broadcast_in_dim"(%a) <{broadcast_dimensions = array<i64: 2, 0, 1>}> : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
    return %456 : tensor<85x180x20xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x85x180xf64>) -> tensor<85x180x20xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    return %0 : tensor<85x180x20xf64>
// CHECK-NEXT:  }
