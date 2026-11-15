// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=fma_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @one_a(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<4xf32>
  %r = stablehlo.fma %one, %arg0, %arg1 : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @one_b(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<4xf32>
  %r = stablehlo.fma %arg0, %one, %arg1 : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @zero_c(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
  %r = stablehlo.fma %arg0, %arg1, %zero : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @neg_one_a(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %neg_one = stablehlo.constant dense<-1.0> : tensor<4xf32>
  %r = stablehlo.fma %neg_one, %arg0, %arg1 : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @neg_one_b(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %neg_one = stablehlo.constant dense<-1.0> : tensor<4xf32>
  %r = stablehlo.fma %arg0, %neg_one, %arg1 : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @other_constant(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %two = stablehlo.constant dense<2.0> : tensor<4xf32>
  %r = stablehlo.fma %two, %arg0, %arg1 : tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @all_constant() -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<4xf32>
  %two = stablehlo.constant dense<2.0> : tensor<4xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
  %r = stablehlo.fma %one, %two, %zero : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK:  func.func @one_a(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @one_b(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @zero_c(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @neg_one_a(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.subtract %arg1, %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @neg_one_b(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.subtract %arg1, %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @other_constant(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.fma %cst, %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @all_constant() -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %0 = stablehlo.fma %cst, %cst_0, %cst_1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }
