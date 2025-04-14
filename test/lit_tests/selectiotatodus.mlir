// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=select_comp_iota_to_dus" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

  func.func @mid1(%arg0 : tensor<31x800xf64>, %arg1 : tensor<31x800xf64>) -> tensor<31x800xf64> {
    %c = stablehlo.constant dense<30> : tensor<31x800xui32>
    %c_0 = stablehlo.constant dense<1> : tensor<31x800xui32>
    %0 = stablehlo.iota dim = 0 : tensor<31x800xui32>
    %1 = stablehlo.compare  GE, %0, %c_0 : (tensor<31x800xui32>, tensor<31x800xui32>) -> tensor<31x800xi1>
    %2 = stablehlo.compare  LT, %0, %c : (tensor<31x800xui32>, tensor<31x800xui32>) -> tensor<31x800xi1>
    %3 = stablehlo.and %1, %2 : tensor<31x800xi1>
    %s = stablehlo.select %3, %arg0, %arg1 : tensor<31x800xi1>, tensor<31x800xf64>
    return %s : tensor<31x800xf64>
  }

// CHECK:  func.func @mid1(%arg0: tensor<31x800xf64>, %arg1: tensor<31x800xf64>) -> tensor<31x800xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [30:31, 0:800] : (tensor<31x800xf64>) -> tensor<1x800xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg1, %0, %c, %c_0 : (tensor<31x800xf64>, tensor<1x800xf64>, tensor<i32>, tensor<i32>) -> tensor<31x800xf64>
// CHECK-NEXT:    return %1 : tensor<31x800xf64>
// CHECK-NEXT:  }