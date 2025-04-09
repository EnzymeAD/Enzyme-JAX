// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" {

  func.func @main(%arg29: tensor<1x24x96xf64>) -> (tensor<1x8x96xf64>) {
      %11831 = stablehlo.slice %arg29 [0:1, 0:8, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %11826 = stablehlo.slice %arg29 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
      %11830 = stablehlo.slice %arg29 [0:1, 0:8, 8:16] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %11835 = stablehlo.concatenate %11831, %11826, %11830, dim = 2 : (tensor<1x8x8xf64>, tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x96xf64>
      stablehlo.return %11835 :  tensor<1x8x96xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x96xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 2 : (tensor<1x8x96xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<1x8x96xf64>
// CHECK-NEXT:  }