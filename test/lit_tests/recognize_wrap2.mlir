// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" {

  func.func @main(%5291: tensor<1x8x80xf64>) -> (tensor<1x8x96xf64>) {
      %11832 = stablehlo.slice %5291 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
      %11828 = stablehlo.slice %5291 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
      %11835 = stablehlo.concatenate %11832, %5291, %11828, dim = 2 : (tensor<1x8x8xf64>, tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x96xf64>
      stablehlo.return %11835 :  tensor<1x8x96xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x96xf64> {
// CHECK-NEXT:    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %1 = stablehlo.concatenate %0, dim = 2 : (tensor<1x8x96xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    stablehlo.return %1 : tensor<1x8x96xf64>
// CHECK-NEXT:  }