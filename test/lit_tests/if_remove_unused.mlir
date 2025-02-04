// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=if_remove_unused" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %7 = stablehlo.compare  GE, %arg0, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8:3 = "stablehlo.if"(%7) ({
      %10 = stablehlo.multiply %arg0, %c2 : tensor<i64>
      stablehlo.return %10, %10, %c2 : tensor<i64>, tensor<i64>, tensor<i64>
    }, {
      %10 = stablehlo.add %arg0, %c3 : tensor<i64>
      stablehlo.return %10, %10, %c3 : tensor<i64>, tensor<i64>, tensor<i64>
    }) : (tensor<i1>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    %9 = stablehlo.add %8#0, %8#2 : tensor<i64>
    return %9 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.compare  GE, %arg0, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:    %1:2 = "stablehlo.if"(%0) ({
// CHECK-NEXT:      %3 = stablehlo.multiply %arg0, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %3, %c_0 : tensor<i64>, tensor<i64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3 = stablehlo.add %arg0, %c_1 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %3, %c_1 : tensor<i64>, tensor<i64>
// CHECK-NEXT:    }) : (tensor<i1>) -> (tensor<i64>, tensor<i64>)
// CHECK-NEXT:    %2 = stablehlo.add %1#0, %1#1 : tensor<i64>
// CHECK-NEXT:    return %2 : tensor<i64>
