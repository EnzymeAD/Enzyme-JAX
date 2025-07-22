// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_to_onedim_dus"  --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main(%arg0: tensor<100000x768xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<768xf32>) -> tensor<100000x768xf32> {
  %0 = stablehlo.reshape %arg1 : (tensor<768xf32>) -> tensor<1x768xf32>
  %1 = stablehlo.slice %arg0 [1:100000, 0:768] : (tensor<100000x768xf32>) -> tensor<99999x768xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x768xf32>, tensor<99999x768xf32>) -> tensor<100000x768xf32>
  return %2 : tensor<100000x768xf32>
}

// CHECK:      func.func @main(%arg0: tensor<100000x768xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<768xf32>) -> tensor<100000x768xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg1 : (tensor<768xf32>) -> tensor<1x768xf32>
// CHECK-NEXT:   %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %c : (tensor<100000x768xf32>, tensor<1x768xf32>, tensor<i64>, tensor<i64>) -> tensor<100000x768xf32>
// CHECK-NEXT:   return %1 : tensor<100000x768xf32>
// CHECK-NEXT: }
