// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-generate-td{patterns=reorder_elementwise_and_shape_op},transform-interpreter,enzyme-hlo-remove-transform)' | FileCheck %s

module {
  func.func @reproduce(%arg0: tensor<1x6x1xcomplex<f64>>) -> tensor<1x3x2xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x6x1xcomplex<f64>>) -> tensor<3x2xcomplex<f64>>
    %1 = stablehlo.real %0 : (tensor<3x2xcomplex<f64>>) -> tensor<3x2xf64>
    %2 = stablehlo.reshape %1 : (tensor<3x2xf64>) -> tensor<1x3x2xf64>
    return %2 : tensor<1x3x2xf64>
  }
}

// CHECK: func.func @reproduce(%arg0: tensor<1x6x1xcomplex<f64>>) -> tensor<1x3x2xf64> {
// CHECK-NEXT: %[[REAL:.*]] = stablehlo.real %arg0 : (tensor<1x6x1xcomplex<f64>>) -> tensor<1x6x1xf64>
// CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[REAL]] : (tensor<1x6x1xf64>) -> tensor<3x2xf64>
// CHECK-NEXT: %[[RESHAPE2:.*]] = stablehlo.reshape %[[RESHAPE1]] : (tensor<3x2xf64>) -> tensor<1x3x2xf64>
// CHECK-NEXT: return %[[RESHAPE2]] : tensor<1x3x2xf64>
// CHECK-NEXT: }
