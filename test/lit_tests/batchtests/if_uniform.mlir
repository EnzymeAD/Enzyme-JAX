// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise | stablehlo-translate - --interpret

module {
  func.func private @cond_scale_or_bias(%x: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %lhs = stablehlo.constant dense<3.0> : tensor<f32>
    %rhs = stablehlo.constant dense<1.0> : tensor<f32>
    %pred = stablehlo.compare GT, %lhs, %rhs : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %res:2 = "stablehlo.if"(%pred) ({
      %two = stablehlo.constant dense<2.0> : tensor<2xf32>
      %three = stablehlo.constant dense<3.0> : tensor<2xf32>
      %a = stablehlo.multiply %x, %two : tensor<2xf32>
      %b = stablehlo.add %x, %three : tensor<2xf32>
      stablehlo.return %a, %b : tensor<2xf32>, tensor<2xf32>
    }, {
      %ten = stablehlo.constant dense<10.0> : tensor<2xf32>
      %neg = stablehlo.negate %x : tensor<2xf32>
      %sub = stablehlo.subtract %x, %ten : tensor<2xf32>
      stablehlo.return %neg, %sub : tensor<2xf32>, tensor<2xf32>
    }) : (tensor<i1>) -> (tensor<2xf32>, tensor<2xf32>)
    return %res#0, %res#1 : tensor<2xf32>, tensor<2xf32>
  }

  func.func @main() {
    %xs = stablehlo.constant dense<[[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]]> : tensor<3x2xf32>
    %rt:2 = enzyme.batch @cond_scale_or_bias(%xs) <{batch_shape = array<i64: 3>}> : (tensor<3x2xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>)
    check.expect_almost_eq_const %rt#0, dense<[[2.0, -4.0], [6.0, 8.0], [-10.0, 12.0]]> : tensor<3x2xf32>
    check.expect_almost_eq_const %rt#1, dense<[[4.0, 1.0], [6.0, 7.0], [-2.0, 9.0]]> : tensor<3x2xf32>
    return
  }
}

// CHECK-LABEL: func.func private @batched_cond_scale_or_bias
// CHECK-NOT: stablehlo.while
// CHECK: "stablehlo.if"
// CHECK-NOT: stablehlo.while
