// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reshape_rotate},transform-interpreter)" | FileCheck %s

module {
  func.func @main(%4949 : tensor<62x62xf64>) -> tensor<1x62x62xf64> {
    %4950 = "enzymexla.rotate"(%4949) <{amount = 61 : i32, dimension = 1 : i32}> : (tensor<62x62xf64>) -> tensor<62x62xf64>
    %4964 = stablehlo.reshape %4950 : (tensor<62x62xf64>) -> tensor<1x62x62xf64>
    return %4964 : tensor<1x62x62xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<62x62xf64>) -> tensor<1x62x62xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<62x62xf64>) -> tensor<1x62x62xf64>
// CHECK-NEXT:    %1 = "enzymexla.rotate"(%0) <{amount = 61 : i32, dimension = 2 : i32}> : (tensor<1x62x62xf64>) -> tensor<1x62x62xf64>
// CHECK-NEXT:    return %1 : tensor<1x62x62xf64>
// CHECK-NEXT:  }
