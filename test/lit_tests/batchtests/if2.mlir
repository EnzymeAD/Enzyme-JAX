// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

module {
  func.func @if_to_select(%result_true_branch: tensor<f32>, %result_false_branch: tensor<f32>) -> tensor<f32> {
    %pred = stablehlo.constant dense<1> : tensor<i1>
    %result:2 = "stablehlo.if"(%pred) ({
      %c = stablehlo.constant dense<1.0> : tensor<f32>
      %1 = stablehlo.add %c, %result_true_branch : tensor<f32>
      "stablehlo.return"(%result_true_branch, %1) : (tensor<f32>, tensor<f32>) -> ()
    }, {
      %c = stablehlo.constant dense<42.0> : tensor<f32>
      %1 = stablehlo.add %c, %result_false_branch : tensor<f32>
      "stablehlo.return"(%result_false_branch, %1) : (tensor<f32>, tensor<f32>) -> ()
    }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
    %0 = stablehlo.add %result#0, %result#1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = enzyme.batch @if_to_select(%arg0, %arg1) {batch_shape = array<i64: 10>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

// CHECK:  func.func private @batched_if_to_select(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<10xi1>
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// CHECK-NEXT:    %0 = stablehlo.add %cst, %arg0 : tensor<10xf32>
// CHECK-NEXT:    %1 = stablehlo.add %arg0, %0 : tensor<10xf32>
// CHECK-NEXT:    return %1 : tensor<10xf32>
// CHECK-NEXT:  }

