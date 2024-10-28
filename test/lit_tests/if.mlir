// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {

  func.func @f(%result_true_branch: tensor<i32>, %result_false_branch: tensor<i32>) -> tensor<i32> {
    %pred = stablehlo.constant dense<1> : tensor<i1>
    %pred2 = stablehlo.constant dense<0> : tensor<i1>

    %result = "stablehlo.if"(%pred) ({
      %c = stablehlo.constant dense<42> : tensor<i32>
      %0 = stablehlo.add %result_true_branch, %c : tensor<i32>

      %result1 = "stablehlo.if"(%pred2) ({
        %c1 = stablehlo.constant dense<42> : tensor<i32>
        %01 = stablehlo.add %0, %c1 : tensor<i32>
        "stablehlo.return"(%01) : (tensor<i32>) -> ()
      }, {
        "stablehlo.return"(%0) : (tensor<i32>) -> ()
      }) : (tensor<i1>) -> tensor<i32>

      "stablehlo.return"(%result1) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>

    return %result : tensor<i32>
  }

}

// CHECK:  func.func @f(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<42> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %c : tensor<i32>
// CHECK-NEXT:    return %0 : tensor<i32>
// CHECK-NEXT:  }
