// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @rotate_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x10x8xf64>, tensor<1x10x7xf64>, tensor<1x10x80xf64>) {
  %0 = stablehlo.slice %arg0 [0:1, 0:10, 0:8] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
  %1 = stablehlo.slice %arg0 [0:1, 0:10, 1:8] : (tensor<1x10x80xf64>) -> tensor<1x10x7xf64>
  %3 = "enzymexla.rotate"(%0) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x10x8xf64>) -> tensor<1x10x8xf64>
  %4 = "enzymexla.rotate"(%1) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x10x7xf64>) -> tensor<1x10x7xf64>
  %5 = "enzymexla.rotate"(%arg0) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
  return %3, %4, %5 : tensor<1x10x8xf64>, tensor<1x10x7xf64>, tensor<1x10x80xf64>
}

// CHECK:      func.func @rotate_operations(%arg0: tensor<1x10x80xf64>, %arg1: tensor<1x10x8xf64>, %arg2: tensor<1x10x8xf64>) -> (tensor<1x10x8xf64>, tensor<1x10x7xf64>, tensor<1x10x80xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.rotate"(%arg0) <{amount = 3 : i32, dimension = 1 : i32}> : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %0 [0:1, 0:10, 1:8] : (tensor<1x10x80xf64>) -> tensor<1x10x7xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %0 [0:1, 0:10, 0:8] : (tensor<1x10x80xf64>) -> tensor<1x10x8xf64>
// CHECK-NEXT:   return %2, %1, %0 : tensor<1x10x8xf64>, tensor<1x10x7xf64>, tensor<1x10x80xf64>
// CHECK-NEXT: }
