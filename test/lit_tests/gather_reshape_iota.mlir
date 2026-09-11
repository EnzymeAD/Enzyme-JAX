// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | stablehlo-translate --interpret --allow-unregistered-dialect
// RUN: stablehlo-translate --interpret --allow-unregistered-dialect %s

// Inserting unit dimensions moves the varying axis from 0 to 1. Each row
// gathers one value, repeated three times: [[1, 1, 1], [3, 3, 3]].
// The add after the reshape keeps it inside the index expression.
func.func @insert_unit_dims(%input: tensor<8xi64>) -> tensor<1x2x3xi64> {
  %two = stablehlo.constant dense<2> : tensor<2x3xi64>
  %one = stablehlo.constant dense<1> : tensor<1x2x1x3xi64>
  %i = stablehlo.iota dim = 0 : tensor<2x3xi64>
  %scaled = stablehlo.multiply %i, %two : tensor<2x3xi64>
  %view = stablehlo.reshape %scaled : (tensor<2x3xi64>) -> tensor<1x2x1x3xi64>
  %indices = stablehlo.add %view, %one : tensor<1x2x1x3xi64>
  %gather = "stablehlo.gather"(%input, %indices) <{
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  }> : (tensor<8xi64>, tensor<1x2x1x3xi64>) -> tensor<1x2x3xi64>
  return %gather : tensor<1x2x3xi64>
}
// CHECK-LABEL: @insert_unit_dims
// CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [1:5:2] : (tensor<8xi64>) -> tensor<2xi64>
// CHECK-NEXT: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [1] : (tensor<2xi64>) -> tensor<1x2x3xi64>
// CHECK-NEXT: return %[[BROADCAST]]

// The non-varying 3x4 dimensions can merge into 12. The varying axis still
// has extent 2 and stride 12, so rows gather 1 and 3 respectively.
func.func @regroup_nonvarying_dims(%input: tensor<8xi64>) -> tensor<2x12xi64> {
  %two = stablehlo.constant dense<2> : tensor<2x3x4xi64>
  %one = stablehlo.constant dense<1> : tensor<2x12x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<2x3x4xi64>
  %scaled = stablehlo.multiply %i, %two : tensor<2x3x4xi64>
  %view = stablehlo.reshape %scaled : (tensor<2x3x4xi64>) -> tensor<2x12x1xi64>
  %indices = stablehlo.add %view, %one : tensor<2x12x1xi64>
  %gather = "stablehlo.gather"(%input, %indices) <{
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  }> : (tensor<8xi64>, tensor<2x12x1xi64>) -> tensor<2x12xi64>
  return %gather : tensor<2x12xi64>
}
// CHECK-LABEL: @regroup_nonvarying_dims
// CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [1:5:2] : (tensor<8xi64>) -> tensor<2xi64>
// CHECK-NEXT: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [0] : (tensor<2xi64>) -> tensor<2x12xi64>
// CHECK-NEXT: return %[[BROADCAST]]

// Equal extents are insufficient: input dimension 1 has stride 3, while
// output dimensions 1 and 2 have strides 2 and 1. Neither is the same iota.
// The result is [[[1,1],[1,3]], [[3,3],[1,1]], [[1,3],[3,3]]].
func.func @changed_stride(%input: tensor<8xi64>) -> tensor<3x2x2xi64> {
  %two = stablehlo.constant dense<2> : tensor<2x2x3xi64>
  %one = stablehlo.constant dense<1> : tensor<3x2x2x1xi64>
  %i = stablehlo.iota dim = 1 : tensor<2x2x3xi64>
  %scaled = stablehlo.multiply %i, %two : tensor<2x2x3xi64>
  %view = stablehlo.reshape %scaled : (tensor<2x2x3xi64>) -> tensor<3x2x2x1xi64>
  %indices = stablehlo.add %view, %one : tensor<3x2x2x1xi64>
  %gather = "stablehlo.gather"(%input, %indices) <{
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  }> : (tensor<8xi64>, tensor<3x2x2x1xi64>) -> tensor<3x2x2xi64>
  return %gather : tensor<3x2x2xi64>
}
// CHECK-LABEL: @changed_stride
// CHECK: %[[GATHER:.*]] = "stablehlo.gather"
// CHECK-NEXT: return %[[GATHER]]

// Splitting the varying axis 6 into 2x3 produces two varying dimensions,
// which cannot be represented by a single IotaLikeTensor.
func.func @split_varying_dim(%input: tensor<16xi64>) -> tensor<2x3xi64> {
  %two = stablehlo.constant dense<2> : tensor<6xi64>
  %one = stablehlo.constant dense<1> : tensor<2x3x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<6xi64>
  %scaled = stablehlo.multiply %i, %two : tensor<6xi64>
  %view = stablehlo.reshape %scaled : (tensor<6xi64>) -> tensor<2x3x1xi64>
  %indices = stablehlo.add %view, %one : tensor<2x3x1xi64>
  %gather = "stablehlo.gather"(%input, %indices) <{
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  }> : (tensor<16xi64>, tensor<2x3x1xi64>) -> tensor<2x3xi64>
  return %gather : tensor<2x3xi64>
}
// CHECK-LABEL: @split_varying_dim
// CHECK: %[[GATHER:.*]] = "stablehlo.gather"
// CHECK-NEXT: return %[[GATHER]]

func.func @main() {
  %input = stablehlo.iota dim = 0 : tensor<8xi64>
  %insert = func.call @insert_unit_dims(%input) : (tensor<8xi64>) -> tensor<1x2x3xi64>
  %insert_expected = stablehlo.constant dense<[[[1, 1, 1], [3, 3, 3]]]> : tensor<1x2x3xi64>
  "check.expect_eq"(%insert, %insert_expected) : (tensor<1x2x3xi64>, tensor<1x2x3xi64>) -> ()
  %regroup = func.call @regroup_nonvarying_dims(%input) : (tensor<8xi64>) -> tensor<2x12xi64>
  %regroup_expected = stablehlo.constant dense<[[1,1,1,1,1,1,1,1,1,1,1,1], [3,3,3,3,3,3,3,3,3,3,3,3]]> : tensor<2x12xi64>
  "check.expect_eq"(%regroup, %regroup_expected) : (tensor<2x12xi64>, tensor<2x12xi64>) -> ()
  %changed = func.call @changed_stride(%input) : (tensor<8xi64>) -> tensor<3x2x2xi64>
  %changed_expected = stablehlo.constant dense<[[[1,1],[1,3]], [[3,3],[1,1]], [[1,3],[3,3]]]> : tensor<3x2x2xi64>
  "check.expect_eq"(%changed, %changed_expected) : (tensor<3x2x2xi64>, tensor<3x2x2xi64>) -> ()
  %input16 = stablehlo.iota dim = 0 : tensor<16xi64>
  %split = func.call @split_varying_dim(%input16) : (tensor<16xi64>) -> tensor<2x3xi64>
  %split_expected = stablehlo.constant dense<[[1,3,5], [7,9,11]]> : tensor<2x3xi64>
  "check.expect_eq"(%split, %split_expected) : (tensor<2x3xi64>, tensor<2x3xi64>) -> ()
  return
}
