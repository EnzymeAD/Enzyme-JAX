// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @dus_f64(%operand: tensor<4x4xf64>, %update: tensor<2x2xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<4x4xf64> {
  %0 = stablehlo.dynamic_update_slice %operand, %update, %i, %j : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %operand = stablehlo.constant dense<1.1> : tensor<4x4xf64>
  %update = stablehlo.constant dense<2.2> : tensor<2x2xf64>
  %i = stablehlo.constant dense<1> : tensor<i32>
  %j = stablehlo.constant dense<1> : tensor<i32>
  
  %expected = stablehlo.constant dense<[[1.1, 1.1, 1.1, 1.1],
                                       [1.1, 2.2, 2.2, 1.1],
                                       [1.1, 2.2, 2.2, 1.1],
                                       [1.1, 1.1, 1.1, 1.1]]> : tensor<4x4xf64>
                                       
  %res = func.call @dus_f64(%operand, %update, %i, %j) : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  return
}
