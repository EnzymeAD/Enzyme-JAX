// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_elementwise<1>},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

func.func @main(%arg0: tensor<16xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
  %27 = stablehlo.convert %arg0 : tensor<16xf64>
  %28 = stablehlo.slice %27 [1:12:5] : (tensor<16xf64>) -> tensor<3xf64>
  %31 = stablehlo.slice %27 [2:13:5] : (tensor<16xf64>) -> tensor<3xf64>
  %34 = stablehlo.slice %27 [4:15:5] : (tensor<16xf64>) -> tensor<3xf64>
  return %28, %31, %34 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
}

// CHECK: func.func @main(%arg0: tensor<16xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:15] : (tensor<16xf64>) -> tensor<14xf64>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : tensor<14xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [3:14:5] : (tensor<14xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %1 [1:12:5] : (tensor<14xf64>) -> tensor<3xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %1 [0:11:5] : (tensor<14xf64>) -> tensor<3xf64>
// CHECK-NEXT:    return %4, %3, %2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT:  }
