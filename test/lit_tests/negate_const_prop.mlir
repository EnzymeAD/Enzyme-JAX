// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test neg(mul(const, var)) should constant-fold the neg(const)
module {
  func.func @neg_mul_const(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %0 = stablehlo.multiply %c, %arg0 : tensor<4x4xf32>
    %1 = stablehlo.negate %0 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @neg_mul_const(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<-2.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[C]], %arg0 : tensor<4x4xf32>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x4xf32>
// CHECK-NEXT:  }

// Test neg(div(const, var)) should constant-fold the neg(const)
module {
  func.func @neg_div_const_lhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %0 = stablehlo.divide %c, %arg0 : tensor<4x4xf32>
    %1 = stablehlo.negate %0 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @neg_div_const_lhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<-2.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:    %[[DIV:.*]] = stablehlo.divide %[[C]], %arg0 : tensor<4x4xf32>
// CHECK-NEXT:    return %[[DIV]] : tensor<4x4xf32>
// CHECK-NEXT:  }

// Test neg(div(var, const)) should constant-fold the neg(const)
module {
  func.func @neg_div_const_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %0 = stablehlo.divide %arg0, %c : tensor<4x4xf32>
    %1 = stablehlo.negate %0 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @neg_div_const_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<-2.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:    %[[DIV:.*]] = stablehlo.divide %arg0, %[[C]] : tensor<4x4xf32>
// CHECK-NEXT:    return %[[DIV]] : tensor<4x4xf32>
// CHECK-NEXT:  }

// Test neg(mul(var, const)) should constant-fold the neg(const)
module {
  func.func @neg_mul_const_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<3.0> : tensor<4x4xf32>
    %0 = stablehlo.multiply %arg0, %c : tensor<4x4xf32>
    %1 = stablehlo.negate %0 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK:  func.func @neg_mul_const_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<-3.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg0, %[[C]] : tensor<4x4xf32>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x4xf32>
// CHECK-NEXT:  }
