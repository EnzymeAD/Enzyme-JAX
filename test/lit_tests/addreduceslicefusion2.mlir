// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=add_reduce_slice_fusion},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<2xui32>) -> tensor<1xui32> {
  %c = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
  %computed = stablehlo.add %arg0, %c : tensor<2xui32>

  %s0 = stablehlo.slice %computed [0:1] : (tensor<2xui32>) -> tensor<1xui32>
  %s1 = stablehlo.slice %computed [1:2] : (tensor<2xui32>) -> tensor<1xui32>

  %s2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>

  %a1 = stablehlo.add %s0, %s1 : tensor<1xui32>
  %a2 = stablehlo.add %s2, %a1 : tensor<1xui32>
  return %a2 : tensor<1xui32>
}


// CHECK:  func.func @main(%arg0: tensor<2xui32>) -> tensor<1xui32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %c_0 : tensor<2xui32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [0:2] : (tensor<2xui32>) -> tensor<2xui32>
// CHECK-NEXT:    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<ui32>) -> tensor<1xui32>
// CHECK-NEXT:    %5 = stablehlo.add %4, %1 : tensor<1xui32>
// CHECK-NEXT:    return %5 : tensor<1xui32>
// CHECK-NEXT:  }
