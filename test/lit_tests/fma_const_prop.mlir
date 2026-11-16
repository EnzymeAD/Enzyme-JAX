// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=fma_const_prop" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @splat() -> tensor<4xf32> {
  %a = stablehlo.constant dense<2.0> : tensor<4xf32>
  %b = stablehlo.constant dense<3.0> : tensor<4xf32>
  %c = stablehlo.constant dense<1.0> : tensor<4xf32>
  %r = stablehlo.fma %a, %b, %c : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @non_splat() -> tensor<4xf64> {
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %b = stablehlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>
  %c = stablehlo.constant dense<[0.5, 1.5, 2.5, 3.5]> : tensor<4xf64>
  %r = stablehlo.fma %a, %b, %c : tensor<4xf64>
  return %r : tensor<4xf64>
}

func.func @mixed_splat() -> tensor<2xf32> {
  %a = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<4.0> : tensor<2xf32>
  %c = stablehlo.constant dense<0.5> : tensor<2xf32>
  %r = stablehlo.fma %a, %b, %c : tensor<2xf32>
  return %r : tensor<2xf32>
}

func.func @single_rounding() -> tensor<f32> {
  %a = stablehlo.constant dense<0x3F800001> : tensor<f32>
  %b = stablehlo.constant dense<0x3F7FFFFE> : tensor<f32>
  %c = stablehlo.constant dense<-1.0> : tensor<f32>
  %r = stablehlo.fma %a, %b, %c : tensor<f32>
  return %r : tensor<f32>
}

func.func @non_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %b = stablehlo.constant dense<3.0> : tensor<4xf32>
  %c = stablehlo.constant dense<1.0> : tensor<4xf32>
  %r = stablehlo.fma %arg0, %b, %c : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK:  func.func @splat() -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<7.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    return %cst : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @non_splat() -> tensor<4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[5.500000e+00, 1.350000e+01, 2.350000e+01, 3.550000e+01]> : tensor<4xf64>
// CHECK-NEXT:    return %cst : tensor<4xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @mixed_splat() -> tensor<2xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[4.500000e+00, 8.500000e+00]> : tensor<2xf32>
// CHECK-NEXT:    return %cst : tensor<2xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @single_rounding() -> tensor<f32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<-1.42108547E-14> : tensor<f32>
// CHECK-NEXT:    return %cst : tensor<f32>
// CHECK-NEXT:  }

// CHECK:  func.func @non_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.fma %arg0, %cst, %cst_0 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }
