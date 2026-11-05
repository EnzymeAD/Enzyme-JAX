// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @relu_const_prop() -> tensor<3xf32> {
    %arg = stablehlo.constant dense<[-1.000000e+00, 0.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %result = enzymexla.math.relu %arg : (tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }

  func.func @gelu_tanh_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.gelu %arg, approximation = <TANH> : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @gelu_sigmoid_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.gelu %arg, approximation = <SIGMOID> : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @gelu_none_not_const_prop() -> tensor<1xf32> {
    %arg = stablehlo.constant dense<[0.000000e+00]> : tensor<1xf32>
    %result = enzymexla.math.gelu %arg, approximation = <NONE> : (tensor<1xf32>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }

  func.func @softplus_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.softplus %arg : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @elu_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.elu %arg, alpha = 1.000000e+00 : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @selu_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.selu %arg : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @celu_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e-06, 1.000000e+03, -1.000000e-06, -1.000000e+03, 2.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.celu %arg, alpha = 2.000000e+00 : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @elu_splat_const_prop() -> tensor<2x3xf32> {
    %arg = stablehlo.constant dense<-1.000000e+00> : tensor<2x3xf32>
    %result = enzymexla.math.elu %arg, alpha = 1.000000e+00 : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  func.func @tgamma_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 5.000000e+00, -1.000000e+00, -3.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.tgamma %arg : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }

  func.func @lgamma_const_prop() -> tensor<5xf32> {
    %arg = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 5.000000e+00, -1.000000e+00, -3.000000e+00]> : tensor<5xf32>
    %result = enzymexla.math.lgamma %arg : (tensor<5xf32>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }
}

// CHECK-LABEL: func.func @relu_const_prop
// CHECK-NOT: enzymexla.math.relu
// CHECK: %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 2.000000e+00]> : tensor<3xf32>
// CHECK-NEXT: return %cst : tensor<3xf32>

// CHECK-LABEL: func.func @gelu_tanh_const_prop
// CHECK-NOT: enzymexla.math.gelu
// CHECK: %cst = stablehlo.constant dense<[5.00000397E-7, 1.000000e+03, -4.99999601E-7, -0.000000e+00, 1.95459771]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @gelu_sigmoid_const_prop
// CHECK-NOT: enzymexla.math.gelu
// CHECK: %cst = stablehlo.constant dense<[5.00000397E-7, 1.000000e+03, -4.99999658E-7, -0.000000e+00, 1.95459783]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @gelu_none_not_const_prop
// CHECK: enzymexla.math.gelu {{.*}}approximation = <NONE>

// CHECK-LABEL: func.func @softplus_const_prop
// CHECK-NOT: enzymexla.math.softplus
// CHECK: %cst = stablehlo.constant dense<[0.693147659, 1.000000e+03, 0.693146646, 0.000000e+00, 2.12692809]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @elu_const_prop
// CHECK-NOT: enzymexla.math.elu
// CHECK: %cst = stablehlo.constant dense<[9.99999997E-7, 1.000000e+03, -9.99999543E-7, -1.000000e+00, 2.000000e+00]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @selu_const_prop
// CHECK-NOT: enzymexla.math.selu
// CHECK: %cst = stablehlo.constant dense<[1.05070103E-6, 1050.70105, -1.75809851E-6, -1.75809932, 2.10140204]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @celu_const_prop
// CHECK-NOT: enzymexla.math.celu
// CHECK: %cst = stablehlo.constant dense<[9.99999997E-7, 1.000000e+03, -9.9999977E-7, -2.000000e+00, 2.000000e+00]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @elu_splat_const_prop
// CHECK-NOT: enzymexla.math.elu
// CHECK: %cst = stablehlo.constant dense<-0.632120549> : tensor<2x3xf32>
// CHECK-NEXT: return %cst : tensor<2x3xf32>

// CHECK-LABEL: func.func @tgamma_const_prop
// CHECK-NOT: enzymexla.math.tgamma
// CHECK: %cst = stablehlo.constant dense<[1.000000e+00, 1.000000e+00, 2.400000e+01, 0x7FC00000, 0x7FC00000]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>

// CHECK-LABEL: func.func @lgamma_const_prop
// CHECK-NOT: enzymexla.math.lgamma
// CHECK: %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 3.17805386, 0x7F800000, 0x7F800000]> : tensor<5xf32>
// CHECK-NEXT: return %cst : tensor<5xf32>
