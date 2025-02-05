// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=if_pred_propagation" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%cond: tensor<i1>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %0 = "stablehlo.if"(%cond) ({
      %1 = "stablehlo.if"( %cond ) ({
        stablehlo.return %c : tensor<i64>
      }, {
        stablehlo.return %c2 : tensor<i64>
      }) : (tensor<i1>) -> tensor<i64>
      stablehlo.return %1 : tensor<i64>
    }, {
      %1 = "stablehlo.if"( %cond ) ({
        stablehlo.return %c : tensor<i64>
      }, {
        stablehlo.return %c2 : tensor<i64>
      }) : (tensor<i1>) -> tensor<i64>
      stablehlo.return %1 : tensor<i64>
    }) : (tensor<i1>) -> tensor<i64>
    %2 = stablehlo.select %cond, %c, %c2 : tensor<i1>, tensor<i64>
    %3 = stablehlo.add %2, %0 : tensor<i64>
    return %3 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i1>) -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %0 = "stablehlo.if"(%arg0) ({
// CHECK-NEXT:      %3 = "stablehlo.if"(%c_2) ({
// CHECK-NEXT:        stablehlo.return %c : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %c_0 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3 = "stablehlo.if"(%c_1) ({
// CHECK-NEXT:        stablehlo.return %c : tensor<i64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %c_0 : tensor<i64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i64>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.select %arg0, %c, %c_0 : tensor<i1>, tensor<i64>
// CHECK-NEXT:    %2 = stablehlo.add %1, %0 : tensor<i64>
// CHECK-NEXT:    return %2 : tensor<i64>
// CHECK-NEXT:  }
