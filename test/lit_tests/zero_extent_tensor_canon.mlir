// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=zero_extent_tensor_canon<16>" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Zero-extent tensors threaded through a while loop and returned from the
// function must be materialized as `stablehlo.constant dense<>` and never as
// `tensor.empty`, which cannot be exported to XLA.

func.func @main(%arg0: tensor<0xf32>, %arg1: tensor<3xf32>) -> (tensor<0xf32>, tensor<3xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<3> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %cst = stablehlo.constant dense<2.000000e+00> : tensor<3xf32>
  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<0xf32>, tensor<3xf32>
   cond {
    %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %2 = stablehlo.add %iterArg_3, %cst : tensor<3xf32>
    stablehlo.return %1, %iterArg_2, %2 : tensor<i64>, tensor<0xf32>, tensor<3xf32>
  }
  return %0#1, %0#2 : tensor<0xf32>, tensor<3xf32>
}

// CHECK-NOT: tensor.empty
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<0xf32>, %arg1: tensor<3xf32>) -> (tensor<0xf32>, tensor<3xf32>)
// CHECK: %[[EMPTY:.+]] = stablehlo.constant dense<> : tensor<0xf32>
// CHECK: %[[WHILE:.+]]:3 = stablehlo.while(%iterArg = %{{.+}}, %iterArg_{{[0-9]+}} = %[[EMPTY]], %iterArg_{{[0-9]+}} = %arg1) : tensor<i64>, tensor<0xf32>, tensor<3xf32>
// CHECK: stablehlo.return %{{.+}}, %[[EMPTY]], %{{.+}} : tensor<i64>, tensor<0xf32>, tensor<3xf32>
// CHECK: return %[[EMPTY]], %[[WHILE]]#2 : tensor<0xf32>, tensor<3xf32>
// CHECK-NOT: tensor.empty

// Zero-extent operands of ops that are not otherwise simplified away (e.g.
// custom calls) must also become constants.

func.func @custom_call(%arg0: tensor<4xf32>, %arg1: tensor<0xf32>) -> tensor<4xf32> {
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) : (tensor<4xf32>, tensor<0xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @custom_call
// CHECK-NOT: tensor.empty
// CHECK: %[[C:.+]] = stablehlo.constant dense<> : tensor<0xf32>
// CHECK: stablehlo.custom_call @foo(%arg0, %[[C]])
// CHECK-NOT: tensor.empty

// Pre-existing `tensor.empty` operands of stablehlo ops are also rewritten
// into exportable constants.

func.func @from_tensor_empty(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<0xf32>
  %1 = stablehlo.custom_call @foo(%arg0, %0) : (tensor<4xf32>, tensor<0xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @from_tensor_empty
// CHECK-NOT: tensor.empty
// CHECK: %[[C:.+]] = stablehlo.constant dense<> : tensor<0xf32>
// CHECK: stablehlo.custom_call @foo(%arg0, %[[C]])
// CHECK-NOT: tensor.empty
