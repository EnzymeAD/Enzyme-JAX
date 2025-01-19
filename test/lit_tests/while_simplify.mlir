// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%x: tensor<f64>) -> tensor<f64> {
    %init_i = stablehlo.constant dense<0> : tensor<i64>
    %init_sum = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %one_f = stablehlo.constant dense<2.0> : tensor<f64>
    %ten = stablehlo.constant dense<3> : tensor<i64>
    %constant = stablehlo.constant dense<42.0> : tensor<f64>
    %results0, %results1, %results2 = "stablehlo.while"(%init_i, %x, %constant) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<f64>, %arg2: tensor<f64>):
      %cond = "stablehlo.compare"(%arg0, %ten) {
        comparison_direction = #stablehlo<comparison_direction LT>
      } : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<f64>, %arg2: tensor<f64>):
      %new_sum = stablehlo.multiply %arg1, %arg2 : tensor<f64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum, %arg2 : tensor<i64>, tensor<f64>, tensor<f64>
    }) : (tensor<i64>, tensor<f64>, tensor<f64>) -> (tensor<i64>, tensor<f64>, tensor<f64>)
    %new_result = stablehlo.add %results1, %results2 : tensor<f64>
    return %new_result : tensor<f64>
  }
}

// CHECK:   func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<4.200000e+01> : tensor<f64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:       %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.multiply %iterArg_2, %cst : tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %3, %2 : tensor<i64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1 = stablehlo.add %0#1, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT:   }
