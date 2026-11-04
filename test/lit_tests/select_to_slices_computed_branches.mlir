// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=select_comp_iota_const_simplify;select_comp_iota_to_dus;select_pad_to_dus" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// The select-to-slices rewrites only cut branches that are cheap to slice
// (arguments, constants, broadcasts, iotas, views of those). Cutting an
// elementwise computation into pieces makes the backend keep the pieces
// apart: a raised WENO stencil's boundary rows went from one XLA kernel to
// six, 4x slower.

// CHECK-LABEL: @iota_compare_computed
// CHECK: stablehlo.select
// CHECK-NOT: stablehlo.concatenate
func.func @iota_compare_computed(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %k = stablehlo.constant dense<3> : tensor<8x4xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %x = stablehlo.multiply %a, %b : tensor<8x4xf32>
  %y = stablehlo.add %a, %b : tensor<8x4xf32>
  %r = stablehlo.select %p, %x, %y : tensor<8x4xi1>, tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// One computed branch is still one cut computation: left alone.

// CHECK-LABEL: @iota_compare_one_computed
// CHECK: stablehlo.select
// CHECK-NOT: stablehlo.concatenate
func.func @iota_compare_one_computed(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %k = stablehlo.constant dense<3> : tensor<8x4xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %x = stablehlo.multiply %a, %b : tensor<8x4xf32>
  %r = stablehlo.select %p, %x, %a : tensor<8x4xi1>, tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// Both branches read from arguments: the rewrite applies as before.

// CHECK-LABEL: @iota_compare_args
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.concatenate
func.func @iota_compare_args(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %k = stablehlo.constant dense<3> : tensor<8x4xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %r = stablehlo.select %p, %a, %b : tensor<8x4xi1>, tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// A uniform predicate (every row is below 8) is still decided, whatever the
// branches are: the select is its true branch.

// CHECK-LABEL: @iota_compare_uniform
// CHECK: %[[X:.+]] = stablehlo.multiply %arg0, %arg1
// CHECK-NEXT: return %[[X]]
func.func @iota_compare_uniform(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %k = stablehlo.constant dense<8> : tensor<8x4xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %x = stablehlo.multiply %a, %b : tensor<8x4xf32>
  %y = stablehlo.add %a, %b : tensor<8x4xf32>
  %r = stablehlo.select %p, %x, %y : tensor<8x4xi1>, tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// The window form (iota >= lo and iota < hi): a computed update over a
// computed base stays a select.

// CHECK-LABEL: @iota_window_computed
// CHECK: stablehlo.select
// CHECK-NOT: stablehlo.dynamic_update_slice
func.func @iota_window_computed(%a: tensor<31x8xf64>, %b: tensor<31x8xf64>) -> tensor<31x8xf64> {
  %c = stablehlo.constant dense<30> : tensor<31x8xui32>
  %c_0 = stablehlo.constant dense<1> : tensor<31x8xui32>
  %0 = stablehlo.iota dim = 0 : tensor<31x8xui32>
  %1 = stablehlo.compare GE, %0, %c_0 : (tensor<31x8xui32>, tensor<31x8xui32>) -> tensor<31x8xi1>
  %2 = stablehlo.compare LT, %0, %c : (tensor<31x8xui32>, tensor<31x8xui32>) -> tensor<31x8xi1>
  %3 = stablehlo.and %1, %2 : tensor<31x8xi1>
  %x = stablehlo.multiply %a, %b : tensor<31x8xf64>
  %y = stablehlo.add %a, %b : tensor<31x8xf64>
  %s = stablehlo.select %3, %x, %y : tensor<31x8xi1>, tensor<31x8xf64>
  return %s : tensor<31x8xf64>
}

// -----

// A computed update written into an argument is an in-place DUS: rewritten.

// CHECK-LABEL: @iota_window_arg_base
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.dynamic_update_slice %arg1
func.func @iota_window_arg_base(%a: tensor<31x8xf64>, %b: tensor<31x8xf64>) -> tensor<31x8xf64> {
  %c = stablehlo.constant dense<30> : tensor<31x8xui32>
  %c_0 = stablehlo.constant dense<1> : tensor<31x8xui32>
  %0 = stablehlo.iota dim = 0 : tensor<31x8xui32>
  %1 = stablehlo.compare GE, %0, %c_0 : (tensor<31x8xui32>, tensor<31x8xui32>) -> tensor<31x8xi1>
  %2 = stablehlo.compare LT, %0, %c : (tensor<31x8xui32>, tensor<31x8xui32>) -> tensor<31x8xi1>
  %3 = stablehlo.and %1, %2 : tensor<31x8xi1>
  %x = stablehlo.multiply %a, %b : tensor<31x8xf64>
  %s = stablehlo.select %3, %x, %b : tensor<31x8xi1>, tensor<31x8xf64>
  return %s : tensor<31x8xf64>
}

// -----

// The padded-constant mask form, computed over computed: stays.

// CHECK-LABEL: @pad_mask_computed
// CHECK: stablehlo.select
// CHECK-NOT: stablehlo.dynamic_update_slice
func.func @pad_mask_computed(%a: tensor<16x8xf64>, %b: tensor<16x8xf64>) -> tensor<16x8xf64> {
  %t = stablehlo.constant dense<true> : tensor<1x8xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.pad %t, %f, low = [0, 0], high = [15, 0], interior = [0, 0] : (tensor<1x8xi1>, tensor<i1>) -> tensor<16x8xi1>
  %x = stablehlo.multiply %a, %b : tensor<16x8xf64>
  %y = stablehlo.add %a, %b : tensor<16x8xf64>
  %s = stablehlo.select %m, %x, %y : tensor<16x8xi1>, tensor<16x8xf64>
  return %s : tensor<16x8xf64>
}

// -----

// CHECK-LABEL: @pad_mask_arg_base
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.dynamic_update_slice %arg1
func.func @pad_mask_arg_base(%a: tensor<16x8xf64>, %b: tensor<16x8xf64>) -> tensor<16x8xf64> {
  %t = stablehlo.constant dense<true> : tensor<1x8xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.pad %t, %f, low = [0, 0], high = [15, 0], interior = [0, 0] : (tensor<1x8xi1>, tensor<i1>) -> tensor<16x8xi1>
  %x = stablehlo.multiply %a, %b : tensor<16x8xf64>
  %s = stablehlo.select %m, %x, %b : tensor<16x8xi1>, tensor<16x8xf64>
  return %s : tensor<16x8xf64>
}

// -----

// A reverse is a view: a halo fill picking between a slice and its reverse
// is still cut into pieces.

// CHECK-LABEL: @reverse_branch
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.concatenate
func.func @reverse_branch(%a: tensor<16xf64>) -> tensor<16xf64> {
  %iota = stablehlo.iota dim = 0 : tensor<16xi32>
  %k = stablehlo.constant dense<8> : tensor<16xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
  %r = stablehlo.reverse %a, dims = [0] : tensor<16xf64>
  %s = stablehlo.select %p, %a, %r : tensor<16xi1>, tensor<16xf64>
  return %s : tensor<16xf64>
}

// -----

// Integer selects are index arithmetic and keep being cut whatever their
// branches.

// CHECK-LABEL: @integer_computed
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.concatenate
func.func @integer_computed(%a: tensor<8x4xi64>, %b: tensor<8x4xi64>) -> tensor<8x4xi64> {
  %iota = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %k = stablehlo.constant dense<3> : tensor<8x4xi32>
  %p = stablehlo.compare LT, %iota, %k : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
  %x = stablehlo.multiply %a, %b : tensor<8x4xi64>
  %y = stablehlo.add %a, %b : tensor<8x4xi64>
  %r = stablehlo.select %p, %x, %y : tensor<8x4xi1>, tensor<8x4xi64>
  return %r : tensor<8x4xi64>
}

// -----

// A gather of an argument is a read of it: cut.

// CHECK-LABEL: @gather_branch
// CHECK-NOT: stablehlo.select
// CHECK: stablehlo.dynamic_update_slice
func.func @gather_branch(%a: tensor<16xf64>, %idx: tensor<16x1xi64>, %b: tensor<16xf64>) -> tensor<16xf64> {
  %t = stablehlo.constant dense<true> : tensor<4xi1>
  %f = stablehlo.constant dense<false> : tensor<i1>
  %m = stablehlo.pad %t, %f, low = [0], high = [12], interior = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<16xi1>
  %g = "stablehlo.gather"(%a, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<16x1xi64>) -> tensor<16xf64>
  %x = stablehlo.multiply %b, %b : tensor<16xf64>
  %s = stablehlo.select %m, %g, %x : tensor<16xi1>, tensor<16xf64>
  return %s : tensor<16xf64>
}
