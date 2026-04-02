// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  // (5 - iota) LT 3  →  iota GT (5 - 3)  →  iota GT 2
  func.func @cst_minus_iota_lt() -> tensor<4xi1> {
    %c5 = stablehlo.constant dense<5> : tensor<4xi64>
    %c3 = stablehlo.constant dense<3> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %sub = stablehlo.subtract %c5, %iota : tensor<4xi64>
    %cmp = stablehlo.compare LT, %sub, %c3, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @cst_minus_iota_lt() -> tensor<4xi1> {
  // CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<[false, false, false, true]> : tensor<4xi1>
  // CHECK-NEXT:    return %[[CST]]
  // CHECK-NEXT:  }

  // (10 - iota) GE 4  →  iota LE (10 - 4)  →  iota LE 6
  func.func @cst_minus_iota_ge() -> tensor<8xi1> {
    %c10 = stablehlo.constant dense<10> : tensor<8xi64>
    %c4  = stablehlo.constant dense<4>  : tensor<8xi64>
    %iota = stablehlo.iota dim = 0 : tensor<8xi64>
    %sub = stablehlo.subtract %c10, %iota : tensor<8xi64>
    %cmp = stablehlo.compare GE, %sub, %c4, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    return %cmp : tensor<8xi1>
  }
  // CHECK:  func.func @cst_minus_iota_ge() -> tensor<8xi1> {
  // CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<[true, true, true, true, true, true, true, false]> : tensor<8xi1>
  // CHECK-NEXT:    return %[[CST]]
  // CHECK-NEXT:  }

  // (7 - iota) EQ 3  →  iota EQ (7 - 3)  →  iota EQ 4
  func.func @cst_minus_iota_eq() -> tensor<8xi1> {
    %c7 = stablehlo.constant dense<7> : tensor<8xi64>
    %c3 = stablehlo.constant dense<3> : tensor<8xi64>
    %iota = stablehlo.iota dim = 0 : tensor<8xi64>
    %sub = stablehlo.subtract %c7, %iota : tensor<8xi64>
    %cmp = stablehlo.compare EQ, %sub, %c3, SIGNED : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    return %cmp : tensor<8xi1>
  }
  // CHECK:  func.func @cst_minus_iota_eq() -> tensor<8xi1> {
  // CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<[false, false, false, false, true, false, false, false]> : tensor<8xi1>
  // CHECK-NEXT:    return %[[CST]]
  // CHECK-NEXT:  }
}
