// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=select_pad_to_dus" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

  func.func @mid1(%arg0 : tensor<1520x3056xf64>, %arg1 : tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %c_291 = stablehlo.constant dense<true> : tensor<1x3056xi1>
    %c_292 = stablehlo.constant dense<false> : tensor<i1>
    %62 = stablehlo.pad %c_291, %c_292, low = [0, 0], high = [1519, 0], interior = [0, 0] : (tensor<1x3056xi1>, tensor<i1>) -> tensor<1520x3056xi1>
    %155 = stablehlo.select %62, %arg0, %arg1 : tensor<1520x3056xi1>, tensor<1520x3056xf64>
    return %155 : tensor<1520x3056xf64>
  }

// CHECK:  func.func @mid1(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg1, %0, %c, %c : (tensor<1520x3056xf64>, tensor<1x3056xf64>, tensor<i32>, tensor<i32>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %1 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

  func.func @mid2(%arg0 : tensor<1520x3056xf64>, %arg1 : tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %c_291 = stablehlo.constant dense<false> : tensor<1x3056xi1>
    %c_292 = stablehlo.constant dense<true> : tensor<i1>
    %62 = stablehlo.pad %c_291, %c_292, low = [0, 0], high = [1519, 0], interior = [0, 0] : (tensor<1x3056xi1>, tensor<i1>) -> tensor<1520x3056xi1>
    %155 = stablehlo.select %62, %arg0, %arg1 : tensor<1520x3056xi1>, tensor<1520x3056xf64>
    return %155 : tensor<1520x3056xf64>
  }

// CHECK:  func.func @mid2(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:1, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %c : (tensor<1520x3056xf64>, tensor<1x3056xf64>, tensor<i32>, tensor<i32>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %1 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

  func.func @mid3(%arg0 : tensor<1520x3056xf64>, %arg1 : tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %c_291 = stablehlo.constant dense<true> : tensor<1x3056xi1>
    %c_292 = stablehlo.constant dense<false> : tensor<i1>
    %62 = stablehlo.pad %c_291, %c_292, low = [1519, 0], high = [0, 0], interior = [0, 0] : (tensor<1x3056xi1>, tensor<i1>) -> tensor<1520x3056xi1>
    %155 = stablehlo.select %62, %arg0, %arg1 : tensor<1520x3056xi1>, tensor<1520x3056xf64>
    return %155 : tensor<1520x3056xf64>
  }

// CHECK:  func.func @mid3(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1519> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg1, %0, %c_0, %c : (tensor<1520x3056xf64>, tensor<1x3056xf64>, tensor<i32>, tensor<i32>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %1 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

  func.func @mid4(%arg0 : tensor<1520x3056xf64>, %arg1 : tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %c_291 = stablehlo.constant dense<false> : tensor<1x3056xi1>
    %c_292 = stablehlo.constant dense<true> : tensor<i1>
    %62 = stablehlo.pad %c_291, %c_292, low = [1519, 0], high = [0, 0], interior = [0, 0] : (tensor<1x3056xi1>, tensor<i1>) -> tensor<1520x3056xi1>
    %155 = stablehlo.select %62, %arg0, %arg1 : tensor<1520x3056xi1>, tensor<1520x3056xf64>
    return %155 : tensor<1520x3056xf64>
  }

// CHECK:  func.func @mid4(%arg0: tensor<1520x3056xf64>, %arg1: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1519> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [1519:1520, 0:3056] : (tensor<1520x3056xf64>) -> tensor<1x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c : (tensor<1520x3056xf64>, tensor<1x3056xf64>, tensor<i32>, tensor<i32>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %1 : tensor<1520x3056xf64>
// CHECK-NEXT:  }