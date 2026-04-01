// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=while_op_induction_replacement},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xi64>, %arg1: tensor<i64>) -> tensor<10xi64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<10xi64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10xi64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
      %2 = stablehlo.add %iterArg_2, %c_1 : tensor<10xi64>
      stablehlo.return %1, %2 : tensor<i64>, tensor<10xi64>
    }
    return %0#1 : tensor<10xi64>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xi64>, %arg1: tensor<i64>) -> tensor<10xi64> {
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:   %1 = stablehlo.add %arg0, %0 : tensor<10xi64>
// CHECK-NEXT:   return %1 : tensor<10xi64>
// CHECK-NEXT: }
