// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func private @sub_inf_inf(%i: tensor<i64>, %j: tensor<i64>) -> tensor<1xi1> {
  %table = stablehlo.constant dense<[1.0, 0x7FF0000000000000]> : tensor<2xf64>
  %x = stablehlo.dynamic_slice %table, %i, sizes = [1] : (tensor<2xf64>, tensor<i64>) -> tensor<1xf64>
  %y = stablehlo.dynamic_slice %table, %j, sizes = [1] : (tensor<2xf64>, tensor<i64>) -> tensor<1xf64>
  %diff = stablehlo.subtract %x, %y : tensor<1xf64>
  %cmp = stablehlo.compare EQ, %diff, %diff : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xi1>
  return %cmp : tensor<1xi1>
}

func.func private @add_pos_neg_inf(%i: tensor<i64>, %j: tensor<i64>) -> tensor<1xi1> {
  %table = stablehlo.constant dense<[1.0, 0x7FF0000000000000, 0xFFF0000000000000]> : tensor<3xf64>
  %x = stablehlo.dynamic_slice %table, %i, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
  %y = stablehlo.dynamic_slice %table, %j, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
  %sum = stablehlo.add %x, %y : tensor<1xf64>
  %cmp = stablehlo.compare EQ, %sum, %sum : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xi1>
  return %cmp : tensor<1xi1>
}

func.func private @sin_plus_cos_nan(%x: tensor<f64>) -> tensor<i1> {
  %s = stablehlo.sine %x : tensor<f64>
  %c = stablehlo.cosine %x : tensor<f64>
  %sum = stablehlo.add %s, %c : tensor<f64>
  %cmp = stablehlo.compare EQ, %sum, %sum : (tensor<f64>, tensor<f64>) -> tensor<i1>
  return %cmp : tensor<i1>
}

func.func @add_one_finite_one_inf(%i: tensor<i64>, %j: tensor<i64>) -> tensor<1xi1> {
  %finite_tab = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %inf_tab = stablehlo.constant dense<[1.0, 0x7FF0000000000000]> : tensor<2xf64>
  %x = stablehlo.dynamic_slice %finite_tab, %i, sizes = [1] : (tensor<2xf64>, tensor<i64>) -> tensor<1xf64>
  %y = stablehlo.dynamic_slice %inf_tab, %j, sizes = [1] : (tensor<2xf64>, tensor<i64>) -> tensor<1xf64>
  %sum = stablehlo.add %x, %y : tensor<1xf64>
  %cmp = stablehlo.compare EQ, %sum, %sum : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xi1>
  return %cmp : tensor<1xi1>
}

func.func @main() {
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %false1 = stablehlo.constant dense<false> : tensor<1xi1>

  %r0 = func.call @sub_inf_inf(%c1, %c1) : (tensor<i64>, tensor<i64>) -> tensor<1xi1>
  "check.expect_eq"(%r0, %false1) : (tensor<1xi1>, tensor<1xi1>) -> ()

  %r1 = func.call @add_pos_neg_inf(%c1, %c2) : (tensor<i64>, tensor<i64>) -> tensor<1xi1>
  "check.expect_eq"(%r1, %false1) : (tensor<1xi1>, tensor<1xi1>) -> ()

  %nan = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
  %false0 = stablehlo.constant dense<false> : tensor<i1>
  %r2 = func.call @sin_plus_cos_nan(%nan) : (tensor<f64>) -> tensor<i1>
  "check.expect_eq"(%r2, %false0) : (tensor<i1>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func.func private @sub_inf_inf(
// CHECK: stablehlo.subtract
// CHECK: stablehlo.compare EQ

// CHECK-LABEL: func.func private @add_pos_neg_inf(
// CHECK: stablehlo.add
// CHECK: stablehlo.compare EQ

// CHECK-LABEL: func.func private @sin_plus_cos_nan(
// CHECK: stablehlo.add
// CHECK: stablehlo.compare EQ

// CHECK-LABEL: func.func @add_one_finite_one_inf(
// CHECK-NEXT: %[[TRUE:.+]] = stablehlo.constant dense<true> : tensor<1xi1>
// CHECK-NEXT: return %[[TRUE]] : tensor<1xi1>
