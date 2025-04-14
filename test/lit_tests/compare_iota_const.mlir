// TODO: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=compare_iota_const_simplify --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN:

module {
  func.func @compare_iota_const_le() -> tensor<4xi1> {
    %c = stablehlo.constant dense<2> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %cmp = stablehlo.compare LE, %iota, %c, SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
  // CHECK:  func.func @compare_iota_const_le() -> tensor<4xi1> {
  // CHECK-NEXT:    %[[TRUE:.+]] = stablehlo.constant dense<true> : tensor<3xi1>
  // CHECK-NEXT:    %[[FALSE:.+]] = stablehlo.constant dense<false> : tensor<1xi1>
  // CHECK-NEXT:    %[[CONC:.+]] = stablehlo.concatenate %[[TRUE]], %[[FALSE]], dim = 0 : (tensor<3xi1>, tensor<1xi1>) -> tensor<4xi1>
  // CHECK-NEXT:    return %[[CONC]] : tensor<4xi1>
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
  // CHECK-NEXT:    %[[TRUE:.+]] = stablehlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT:    %[[FALSE:.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %[[TRUE]], %[[FALSE]], low = [2], high = [1], interior = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<4xi1>
  // CHECK-NEXT:    return %[[PAD]] : tensor<4xi1>
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
  // CHECK-NEXT:    %[[FALSE:.+]] = stablehlo.constant dense<false> : tensor<1xi1>
  // CHECK-NEXT:    %[[TRUE:.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-NEXT:    %[[PAD:.+]] = stablehlo.pad %c, %c_0, low = [2], high = [1], interior = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<4xi1>
  // CHECK-NEXT:    return %[[PAD]] : tensor<4xi1>
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
