// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_to_broadcast" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s


  func.func @f(%A: tensor<144x1024x1025xf64>) -> (tensor<1x144x1024x1025xf64>)  {
    %res = stablehlo.reshape %A : (tensor<144x1024x1025xf64>) -> tensor<1x144x1024x1025xf64>
    func.return %res : tensor<1x144x1024x1025xf64>
  }

// CHECK:  func.func @f(%arg0: tensor<144x1024x1025xf64>) -> tensor<1x144x1024x1025xf64> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2, 3] : (tensor<144x1024x1025xf64>) -> tensor<1x144x1024x1025xf64>
// CHECK-NEXT:    return %0 : tensor<1x144x1024x1025xf64>
// CHECK-NEXT:  }