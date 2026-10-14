// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=case_remove_unused" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<i64>, %idx: tensor<i32>) -> tensor<i64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %8:3 = "stablehlo.case"(%idx) ({
      %10 = stablehlo.multiply %arg0, %c2 : tensor<i64>
      stablehlo.return %10, %10, %c2 : tensor<i64>, tensor<i64>, tensor<i64>
    }, {
      %10 = stablehlo.add %arg0, %c3 : tensor<i64>
      stablehlo.return %10, %10, %c3 : tensor<i64>, tensor<i64>, tensor<i64>
    }, {
      stablehlo.return %c, %c2, %c3 : tensor<i64>, tensor<i64>, tensor<i64>
    }) : (tensor<i32>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    %9 = stablehlo.add %8#0, %8#2 : tensor<i64>
    return %9 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i32>) -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %0:2 = "stablehlo.case"(%arg1) ({
// CHECK-NEXT:      %2 = stablehlo.multiply %arg0, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %2, %c_0 : tensor<i64>, tensor<i64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %2 = stablehlo.add %arg0, %c_1 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %2, %c_1 : tensor<i64>, tensor<i64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %c, %c_1 : tensor<i64>, tensor<i64>
// CHECK-NEXT:    }) : (tensor<i32>) -> (tensor<i64>, tensor<i64>)
// CHECK-NEXT:    %1 = stablehlo.add %0#0, %0#1 : tensor<i64>
// CHECK-NEXT:    return %1 : tensor<i64>
