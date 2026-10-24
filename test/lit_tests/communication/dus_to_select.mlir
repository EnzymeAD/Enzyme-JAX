// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_select_comm=1 dus_to_pad_comm=0 rotate_to_pad_comm=0 concat_to_pad_comm=0 extend_to_pad_comm=0 wrap_to_pad_comm=0 updatewithoutcorners_to_select=0 reorder_associative=0 dus_of_concat_slices=0})" --split-input-file %s | FileCheck %s

// A stencil narrowed to the interior of a dimension sharded over two
// devices: the window [1, 14) of 16 is not on the shard boundary (8), so the
// DUS and the two slices of the arguments it was narrowed to each exchange
// halos. The update is recomputed at the full extent from the widened
// argument slices and the window becomes an iota mask.
sdy.mesh @mesh = <["x"=2]>
func.func @interior(%a: tensor<20xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<20xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %base = stablehlo.slice %a [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
  %l = stablehlo.slice %a [3:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<13xf32>
  %r = stablehlo.slice %b [4:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<13xf32>
  %two = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} dense<2.000000e+00> : tensor<13xf32>
  %s = stablehlo.add %l, %r {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %u = stablehlo.multiply %s, %two {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %base, %u, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @interior
// CHECK-NOT: dynamic_update_slice
// CHECK-DAG: %[[BASE:.+]] = stablehlo.slice %arg0 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[L:.+]] = stablehlo.slice %arg0 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[R:.+]] = stablehlo.slice %arg1 [3:19] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[TWO:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} dense<2.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[S:.+]] = stablehlo.add %[[L]], %[[R]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[U:.+]] = stablehlo.multiply %[[S]], %[[TWO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[IOTA:.+]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xi32>
// CHECK-DAG: %[[GE:.+]] = stablehlo.compare GE, %[[IOTA]], %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
// CHECK-DAG: %[[LT:.+]] = stablehlo.compare LT, %[[IOTA]], %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
// CHECK: %[[M:.+]] = stablehlo.and %[[GE]], %[[LT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xi1>
// CHECK-NEXT: %[[SEL:.+]] = stablehlo.select %[[M]], %[[U]], %[[BASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT: return %[[SEL]]

// -----

// A window on the shard boundary needs no exchange and stays a DUS.
sdy.mesh @mesh = <["x"=2]>
func.func @aligned(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c8 = stablehlo.constant dense<8> : tensor<i32>
  %l = stablehlo.slice %b [8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<8xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<8xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @aligned
// CHECK: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.select

// -----

// An unsharded dimension is left to the slices.
sdy.mesh @mesh = <["x"=2]>
func.func @unsharded_dim(%a: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %b: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x16xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %l = stablehlo.slice %b [1:7, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>) -> tensor<6x16xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<6x16xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c1, %c0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>, tensor<6x16xf32>, tensor<i32>, tensor<i32>) -> tensor<8x16xf32>
  return %d : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @unsharded_dim
// CHECK: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.select

// -----

// Nested windows, the shape of a boundary-adapted stencil: the outer update
// is itself a DUS of a narrower one. Both become selects over computations
// at the full extent.
sdy.mesh @mesh = <["x"=2]>
func.func @nested(%a: tensor<20xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<20xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %base = stablehlo.slice %a [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
  %mid = stablehlo.slice %b [3:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<14xf32>
  %midn = stablehlo.negate %mid {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<14xf32>
  %inl = stablehlo.slice %midn [0:12] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<14xf32>) -> tensor<12xf32>
  %inr = stablehlo.slice %midn [2:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<14xf32>) -> tensor<12xf32>
  %in = stablehlo.add %inl, %inr {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<12xf32>
  %d0 = stablehlo.dynamic_update_slice %midn, %in, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<14xf32>, tensor<12xf32>, tensor<i32>) -> tensor<14xf32>
  %d1 = stablehlo.dynamic_update_slice %base, %d0, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<14xf32>, tensor<i32>) -> tensor<16xf32>
  return %d1 : tensor<16xf32>
}

// CHECK-LABEL: func.func @nested
// CHECK-NOT: dynamic_update_slice
// CHECK-DAG: %[[MID:.+]] = stablehlo.slice %arg1 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[MIDN:.+]] = stablehlo.negate %[[MID]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[INL:.+]] = stablehlo.slice %arg1 [1:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[INR:.+]] = stablehlo.slice %arg1 [3:19] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: stablehlo.negate %[[INL]]
// CHECK-DAG: stablehlo.negate %[[INR]]
// CHECK: stablehlo.select
// CHECK: stablehlo.select
// CHECK-NOT: dynamic_update_slice

// -----

// An update computed from an argument that only exists at the window's
// extent cannot be widened; the DUS stays.
sdy.mesh @mesh = <["x"=2]>
func.func @narrow_argument(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %u: tensor<13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %n = stablehlo.negate %u {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %a, %n, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @narrow_argument
// CHECK: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.select
