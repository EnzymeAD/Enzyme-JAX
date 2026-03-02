// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_reshape" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


func.func @r(%10: tensor<1x1520x1xf64>)  -> tensor<1520x1520x2xf64> {
  %11 = stablehlo.reshape %10 : (tensor<1x1520x1xf64>) -> tensor<1520xf64>
  %25 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<1520xf64>) -> tensor<1520x1520x2xf64>
  return %25 : tensor<1520x1520x2xf64>
}

// CHECK:  func.func @r(%arg0: tensor<1x1520x1xf64>) -> tensor<1520x1520x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1520x1xf64>) -> tensor<1520x1520x2xf64>
// CHECK-NEXT:    return %0 : tensor<1520x1520x2xf64>
// CHECK-NEXT:  }

