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

  func.func @if_to_select(%result_true_branch: tensor<i32>, %result_false_branch: tensor<i32>, %pred: tensor<i1>) -> tensor<i32> {
    %result = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
    return %result : tensor<i32>
  }

  func.func @if_to_select2(%result_true_branch: tensor<i32>, %result_false_branch: tensor<i32>, %pred: tensor<i1>) -> tensor<i32> {
    %result:2 = "stablehlo.if"(%pred) ({
      %c = stablehlo.constant dense<1> : tensor<i32>
      %1 = stablehlo.add %c, %result_true_branch : tensor<i32>
      "stablehlo.return"(%result_true_branch, %1) : (tensor<i32>, tensor<i32>) -> ()
    }, {
      %c = stablehlo.constant dense<2> : tensor<i32>
      %1 = stablehlo.add %c, %result_false_branch : tensor<i32>
      "stablehlo.return"(%result_false_branch, %1) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i1>) -> (tensor<i32>, tensor<i32>)
    %0 = stablehlo.add %result#0, %result#1 : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @slice_if_multi_result(%x: tensor<2xf64>, %t: tensor<f64>, %pred: tensor<i1>) -> (tensor<f64>, tensor<1xf64>) {
    %res:2 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%t, %x) : (tensor<f64>, tensor<2xf64>) -> ()
    }, {
      %nt = stablehlo.negate %t : tensor<f64>
      %nx = stablehlo.negate %x : tensor<2xf64>
      "stablehlo.return"(%nt, %nx) : (tensor<f64>, tensor<2xf64>) -> ()
    }) : (tensor<i1>) -> (tensor<f64>, tensor<2xf64>)
    %early = stablehlo.add %res#0, %t : tensor<f64>
    %sliced = stablehlo.slice %res#1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    return %early, %sliced : tensor<f64>, tensor<1xf64>
  }
}

// CHECK:  func.func @f(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<42> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %c : tensor<i32>
// CHECK-NEXT:    return %0 : tensor<i32>
// CHECK-NEXT:  }

// CHECK:  func.func @if_to_select(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i1>) -> tensor<i32> {
// CHECK-NEXT:    %0 = stablehlo.select %arg2, %arg0, %arg1 : tensor<i1>, tensor<i32>
// CHECK-NEXT:    return %0 : tensor<i32>
// CHECK-NEXT:  }

// CHECK:  func.func @if_to_select2(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i1>) -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.select %arg2, %arg0, %arg1 : tensor<i1>, tensor<i32>
// CHECK-NEXT:    %1 = "stablehlo.if"(%arg2) ({
// CHECK-NEXT:      %3 = stablehlo.add %c_0, %arg0 : tensor<i32>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i32>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3 = stablehlo.add %c, %arg1 : tensor<i32>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i32>
// CHECK-NEXT:    }) : (tensor<i1>) -> tensor<i32>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 : tensor<i32>
// CHECK-NEXT:    return %2 : tensor<i32>
// CHECK-NEXT:  }

// CHECK:  func.func @slice_if_multi_result(%arg0: tensor<2xf64>, %arg1: tensor<f64>, %arg2: tensor<i1>) -> (tensor<f64>, tensor<1xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %1:2 = "stablehlo.if"(%arg2) ({
// CHECK-NEXT:      stablehlo.return %arg1, %0 : tensor<f64>, tensor<1xf64>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3 = stablehlo.negate %arg1 : tensor<f64>
// CHECK-NEXT:      %4 = stablehlo.negate %0 : tensor<1xf64>
// CHECK-NEXT:      stablehlo.return %3, %4 : tensor<f64>, tensor<1xf64>
// CHECK-NEXT:    }) : (tensor<i1>) -> (tensor<f64>, tensor<1xf64>)
// CHECK-NEXT:    %2 = stablehlo.add %1#0, %arg1 : tensor<f64>
// CHECK-NEXT:    return %2, %1#1 : tensor<f64>, tensor<1xf64>
// CHECK-NEXT:  }
