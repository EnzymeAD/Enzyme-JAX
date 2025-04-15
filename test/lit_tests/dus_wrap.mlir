// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_concat" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

  func.func @main(%iterArg_2 : tensor<1x39x48xf32>, %6 : tensor<1x24x48xf32>) -> tensor<1x39x62xf32> {
      %c = stablehlo.constant dense<1> : tensor<i64>
      %c_0 = stablehlo.constant dense<7> : tensor<i64>
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %5 = "enzymexla.wrap"(%iterArg_2) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1x39x48xf32>) -> tensor<1x39x62xf32>
      %7 = stablehlo.dynamic_update_slice %5, %6, %c_1, %c_0, %c_0 : (tensor<1x39x62xf32>, tensor<1x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x39x62xf32>
      return %7 : tensor<1x39x62xf32>
  }

// CHECK:    func.func @main(%arg0: tensor<1x39x48xf32>, %arg1: tensor<1x24x48xf32>) -> tensor<1x39x62xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:39, 41:48] : (tensor<1x39x48xf32>) -> tensor<1x39x7xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:39, 0:7] : (tensor<1x39x48xf32>) -> tensor<1x39x7xf32>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_0, %c, %c_0 : (tensor<1x39x48xf32>, tensor<1x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x39x48xf32>
// CHECK-NEXT:    %3 = stablehlo.concatenate %0, %2, %1, dim = 2 : (tensor<1x39x7xf32>, tensor<1x39x48xf32>, tensor<1x39x7xf32>) -> tensor<1x39x62xf32>
// CHECK-NEXT:    return %3 : tensor<1x39x62xf32>
// CHECK-NEXT:  }