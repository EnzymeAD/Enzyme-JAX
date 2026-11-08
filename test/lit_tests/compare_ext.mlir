// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    %c = stablehlo.constant dense<0> : tensor<3xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<3xi64>
    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xi64>
    %1 = stablehlo.compare  EQ, %0, %c,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %2 = stablehlo.compare  EQ, %0, %c_1,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %3 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
    %4 = stablehlo.compare  EQ, %0, %3,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %1, %2, %4 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
  }

  func.func @not_eq(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> tensor<3xi1> {
    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xi64>
    %1 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
    %2 = stablehlo.compare  NE, %0, %1,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %2 : tensor<3xi1>
  }

  func.func @not_bool_const(%arg0: tensor<3xi1>) -> tensor<3xi1> {
    %c = stablehlo.constant dense<2> : tensor<3xi64>
    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xi64>
    %1 = stablehlo.compare  EQ, %0, %c,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %1 : tensor<3xi1>
  }

  func.func @not_from_bool(%arg0: tensor<3xi8>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>) {
    %c = stablehlo.constant dense<1> : tensor<3xi64>
    %0 = stablehlo.convert %arg0 : (tensor<3xi8>) -> tensor<3xi64>
    %1 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
    %2 = stablehlo.compare  EQ, %0, %c,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %3 = stablehlo.compare  EQ, %0, %1,  UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    return %2, %3 : tensor<3xi1>, tensor<3xi1>
  }

  func.func @float(%arg0: tensor<3xi1>) -> tensor<3xi1> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xf32>
    %1 = stablehlo.compare  EQ, %0, %cst,  FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    return %1 : tensor<3xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
// CHECK-NEXT:    %0 = stablehlo.not %arg0 : tensor<3xi1>
// CHECK-NEXT:    %1 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
// CHECK-NEXT:    return %0, %arg0, %1 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
// CHECK-NEXT:  }

// CHECK:  func.func @not_eq(%arg0: tensor<3xi1>, %arg1: tensor<3xi1>) -> tensor<3xi1> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.compare NE, %0, %1, UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    return %2 : tensor<3xi1>
// CHECK-NEXT:  }

// CHECK:  func.func @not_bool_const(%arg0: tensor<3xi1>) -> tensor<3xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<3xi64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3xi1>) -> tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.compare EQ, %0, %c, UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    return %1 : tensor<3xi1>
// CHECK-NEXT:  }

// CHECK:  func.func @not_from_bool(%arg0: tensor<3xi8>, %arg1: tensor<3xi1>) -> (tensor<3xi1>, tensor<3xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3xi8>) -> tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.compare EQ, %0, %c, UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    %3 = stablehlo.compare EQ, %0, %1, UNSIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    return %2, %3 : tensor<3xi1>, tensor<3xi1>
// CHECK-NEXT:  }

// CHECK:  func.func @float(%arg0: tensor<3xi1>) -> tensor<3xi1> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3xi1>) -> tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.compare EQ, %0, %cst, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
// CHECK-NEXT:    return %1 : tensor<3xi1>
// CHECK-NEXT:  }
