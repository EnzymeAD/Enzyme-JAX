// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @fuse(%arg0: tensor<1x35x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<1x20x45xf64>, %i1: tensor<i64>, %i2: tensor<i64>, %i3: tensor<i64>) -> tensor<1x35x59xf64> {  
      %2730 = stablehlo.dynamic_update_slice %arg0, %arg1, %i1, %i2, %i3 : (tensor<1x35x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x35x59xf64>
      %2738 = stablehlo.dynamic_update_slice %2730, %arg2, %i1, %i2, %i3 : (tensor<1x35x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x35x59xf64>
      func.return %2738 : tensor<1x35x59xf64>
  }
}

// CHECK:  func.func @fuse(%arg0: tensor<1x35x59xf64>, %arg1: tensor<1x20x45xf64>, %arg2: tensor<1x20x45xf64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>) -> tensor<1x35x59xf64> {
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg0, %arg2, %arg3, %arg4, %arg5 : (tensor<1x35x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x35x59xf64>
// CHECK-NEXT:    return %0 : tensor<1x35x59xf64>
// CHECK-NEXT:  }