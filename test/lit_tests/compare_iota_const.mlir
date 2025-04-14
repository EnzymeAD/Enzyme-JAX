// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @compare_iota_const_le() -> tensor<4xi1> {
    %c = stablehlo.constant dense<2> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_le() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<[true, true, true, false]> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }

  func.func @compare_iota_const_le_oob() -> tensor<4xi1> {
    %c = stablehlo.constant dense<-1> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_le_oob() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }

  func.func @compare_iota_const_eq() -> tensor<4xi1> {
    %c = stablehlo.constant dense<2> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare EQ, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_eq() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<[false, false, true, false]> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }

  func.func @compare_iota_const_eq_oob() -> tensor<4xi1> {
    %c = stablehlo.constant dense<6> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare EQ, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_eq_oob() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }

  func.func @compare_iota_const_ne() -> tensor<4xi1> {
    %c = stablehlo.constant dense<2> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare NE, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_ne() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<[true, true, false, true]> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }

  func.func @compare_iota_const_ne_oob() -> tensor<4xi1> {
    %c = stablehlo.constant dense<6> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare NE, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_ne_oob() -> tensor<4xi1> {
  // CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<4xi1>
  // CHECK-NEXT:    return %c : tensor<4xi1>
  // CHECK-NEXT:  }
}
