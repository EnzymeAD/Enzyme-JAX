// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=lower_multipad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>

func.func @main(%arg0: tensor<1519x3056xf64>, %cst: tensor<f64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>) {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1519x3056xf64>
  // CHECK-SAME: %[[CST:.*]]: tensor<f64>

  %0:2 = "enzymexla.multi_pad"(%arg0, %cst) <{amount = 1 : i64, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} : (tensor<1519x3056xf64>, tensor<f64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>)

  // CHECK: %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [1, 0], high = [0, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{{.*}}]>]>}
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{{.*}}]>]>}

  // We are returning them, so we just check if they are returned!
  // Wait, in my test I was returning them, let's make sure the output types match!
  // The output of func is (tensor<1521x3056xf64>, tensor<1521x3056xf64>) but MultiPad result is 1520!
  // Let's fix output types of func to match MultiPad results!
  return %0#0, %0#1 : tensor<1520x3056xf64>, tensor<1520x3056xf64>
}
