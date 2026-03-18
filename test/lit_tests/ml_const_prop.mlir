// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @relu_const_prop() -> tensor<3xf32> {
    %arg = stablehlo.constant dense<[-1.000000e+00, 0.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %result = enzymexla.ml.relu %arg : (tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }

  func.func @gelu_tanh_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.ml.gelu %arg, approximation = TANH : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @gelu_sigmoid_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.ml.gelu %arg, approximation = SIGMOID : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @gelu_none_not_const_prop() -> tensor<1xf32> {
    %arg = stablehlo.constant dense<[0.000000e+00]> : tensor<1xf32>
    %result = enzymexla.ml.gelu %arg, approximation = NONE : (tensor<1xf32>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }

  func.func @softplus_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.ml.softplus %arg : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }
}

// CHECK-LABEL: func.func @relu_const_prop
// CHECK-NOT: enzymexla.ml.relu
// CHECK: %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 2.000000e+00]> : tensor<3xf32>
// CHECK-NEXT: return %cst : tensor<3xf32>

// CHECK-LABEL: func.func @gelu_tanh_const_prop
// CHECK-NOT: enzymexla.ml.gelu
// CHECK: %cst = stablehlo.constant dense<[5.00000397E-7, 1.000000e+03, -4.99999601E-7, -0.000000e+00, 1.95459771]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @gelu_sigmoid_const_prop
// CHECK-NOT: enzymexla.ml.gelu
// CHECK: %cst = stablehlo.constant dense<[5.00000397E-7, 1.000000e+03, -4.99999658E-7, -0.000000e+00, 1.95459783]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @gelu_none_not_const_prop
// CHECK: enzymexla.ml.gelu {{.*}}approximation = NONE

// CHECK-LABEL: func.func @softplus_const_prop
// CHECK-NOT: enzymexla.ml.softplus
// CHECK: %cst = stablehlo.constant dense<[0.693147659, 1.000000e+03, 0.693146646, 0.000000e+00, 2.12692809]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>
