// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=real_complex_mul_simplify" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s
func.func @main(%arg0: tensor<4368xf64> , %arg1: tensor<4368xcomplex<f64>> ) -> tensor<4368xcomplex<f64>> {
    %0 = stablehlo.convert %arg0 : (tensor<4368xf64>) -> tensor<4368xcomplex<f64>>
    %1 = stablehlo.multiply %0, %arg1 : tensor<4368xcomplex<f64>>
    return %1 : tensor<4368xcomplex<f64>>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4368xf64>, %[[ARG1:.*]]: tensor<4368xcomplex<f64>>)

// CHECK-DAG: %[[REAL:.*]] = stablehlo.real %[[ARG1]]
// CHECK-DAG: %[[IMAG:.*]] = stablehlo.imag %[[ARG1]]
// CHECK-DAG: %[[MUL_REAL:.*]] = stablehlo.multiply %[[ARG0]], %[[REAL]]
// CHECK-DAG: %[[MUL_IMAG:.*]] = stablehlo.multiply %[[ARG0]], %[[IMAG]]

// CHECK:     %[[RESULT:.*]] = stablehlo.complex %[[MUL_REAL]], %[[MUL_IMAG]]
// CHECK:     return %[[RESULT]]