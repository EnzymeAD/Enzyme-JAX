// RUN: enzymexlamlir-opt --enzyme-hlo-opt --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


func.func @r(%5017: tensor<1xf32>) -> tensor<1x3056xf32> {
  %5028 = stablehlo.broadcast_in_dim %5017, dims = [1] : (tensor<1xf32>) -> tensor<1x3056xf32> 
  return %5028 : tensor<1x3056xf32>
}

// CHECK:  func.func @r(%arg0: tensor<1xf32>) -> tensor<1x3056xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<1x3056xf32>
// CHECK-NEXT:    return %0 : tensor<1x3056xf32>
// CHECK-NEXT:  }

