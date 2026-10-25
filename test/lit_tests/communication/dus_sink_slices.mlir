// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_sink_slices_comm=1 dus_to_pad_comm=0 rotate_to_pad_comm=0 concat_to_pad_comm=0 extend_to_pad_comm=0 wrap_to_pad_comm=0 updatewithoutcorners_to_select=0 reorder_associative=0 dus_of_concat_slices=0})" --split-input-file %s | FileCheck %s

// A stencil narrowed to the interior of a dimension sharded over two
// devices: the window [1, 14) of 16 is not on the shard boundary (8), so the
// DUS and the two slices of the arguments it was narrowed to each exchange
// halos. The slices move down to the DUS: the update is recomputed at the
// full extent (the widened slice of %a is the existing %base) and the DUS
// gets a slice of it at its own window, which the partitioner handles
// without moving data.
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
// CHECK-DAG: %[[C1:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[BASE:.+]] = stablehlo.slice %arg0 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[R:.+]] = stablehlo.slice %arg1 [3:19] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[TWO:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} dense<2.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[S:.+]] = stablehlo.add %[[BASE]], %[[R]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[U:.+]] = stablehlo.multiply %[[S]], %[[TWO]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK: %[[W:.+]] = stablehlo.slice %[[U]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %[[BASE]], %[[W]], %[[C1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
// CHECK-NEXT: return %[[D]]

// -----

// An update that already is a slice at the DUS's window: nothing to move.
sdy.mesh @mesh = <["x"=2]>
func.func @direct_slice(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %u = stablehlo.slice %b [3:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @direct_slice
// CHECK: %[[U:.+]] = stablehlo.slice %arg1 [3:16]
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg0, %[[U]]
// CHECK-NEXT: return %[[D]]

// -----

// A window covering one whole shard. The narrow intermediate is still
// sharded over both devices, so computing it moves half of it; computing the
// negate at the full extent (twice the work, within the default bound) does
// not.
sdy.mesh @mesh = <["x"=2]>
func.func @half_window(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c8 = stablehlo.constant dense<8> : tensor<i32>
  %l = stablehlo.slice %b [8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<8xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<8xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @half_window
// CHECK: %[[N:.+]] = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[N]] [8:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<8xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg0, %[[W]], %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<8xf32>, tensor<i32>) -> tensor<16xf32>
// CHECK-NEXT: return %[[D]]

// -----

// A window of a quarter of the dimension would triple the work: left alone.
sdy.mesh @mesh = <["x"=2]>
func.func @too_much_extra_work(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c6 = stablehlo.constant dense<6> : tensor<i32>
  %l = stablehlo.slice %b [6:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<4xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<4xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @too_much_extra_work
// CHECK: %[[N:.+]] = stablehlo.negate %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
// CHECK-NEXT: stablehlo.dynamic_update_slice %arg0, %[[N]]

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
// CHECK: %[[N:.+]] = stablehlo.negate %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<6x16xf32>
// CHECK-NEXT: stablehlo.dynamic_update_slice %arg0, %[[N]]

// -----

// A narrow intermediate with another user: the other user gets a slice of
// the wide value, so the square is computed once, at the full extent.
sdy.mesh @mesh = <["x"=2]>
func.func @shared_intermediate(%a: tensor<20xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<16xf32>, tensor<13xf32>) {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %base = stablehlo.slice %a [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
  %l = stablehlo.slice %a [3:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<13xf32>
  %g = stablehlo.multiply %l, %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %u = stablehlo.negate %g {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %base, %u, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d, %g : tensor<16xf32>, tensor<13xf32>
}

// CHECK-LABEL: func.func @shared_intermediate
// CHECK-DAG: %[[BASE:.+]] = stablehlo.slice %arg0 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[G:.+]] = stablehlo.multiply %[[BASE]], %[[BASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[GN:.+]] = stablehlo.slice %[[G]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK-DAG: %[[U:.+]] = stablehlo.negate %[[G]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[W:.+]] = stablehlo.slice %[[U]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK: %[[D:.+]] = stablehlo.dynamic_update_slice %[[BASE]], %[[W]]
// CHECK-NEXT: return %[[D]], %[[GN]]
// CHECK-NOT: stablehlo.multiply

// -----

// The wide computation already exists: it is reused rather than rebuilt.
sdy.mesh @mesh = <["x"=2]>
func.func @reuse_existing(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<16xf32>, tensor<16xf32>) {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %gw = stablehlo.multiply %b, %b {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
  %l = stablehlo.slice %b [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
  %g = stablehlo.multiply %l, %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %a, %g, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d, %gw : tensor<16xf32>, tensor<16xf32>
}

// CHECK-LABEL: func.func @reuse_existing
// CHECK: %[[GW:.+]] = stablehlo.multiply %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-NOT: stablehlo.multiply
// CHECK: %[[W:.+]] = stablehlo.slice %[[GW]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg0, %[[W]]
// CHECK-NEXT: return %[[D]], %[[GW]]

// -----

// Nested windows, the shape of a boundary-adapted stencil: the outer update
// is itself a DUS of a narrower one. Both updates end up computed at the
// full extent; each DUS gets the slice at its window.
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
// CHECK-DAG: %[[MID:.+]] = stablehlo.slice %arg1 [2:18] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[MIDN:.+]] = stablehlo.negate %[[MID]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-DAG: %[[INL:.+]] = stablehlo.slice %arg1 [1:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[INR:.+]] = stablehlo.slice %arg1 [3:19] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<20xf32>) -> tensor<16xf32>
// CHECK-DAG: stablehlo.negate %[[INL]]
// CHECK-DAG: stablehlo.negate %[[INR]]
// CHECK: %[[IN:.+]] = stablehlo.slice %{{.+}} [2:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<12xf32>
// CHECK-NEXT: %[[D0:.+]] = stablehlo.dynamic_update_slice %[[MIDN]], %[[IN]], %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<12xf32>, tensor<i32>) -> tensor<16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[D0]] [1:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<14xf32>
// CHECK-NEXT: %[[D1:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %[[W]], %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<14xf32>, tensor<i32>) -> tensor<16xf32>
// CHECK-NEXT: return %[[D1]]

// -----

// Two sharded dimensions narrowed: both are widened; the DUS gets the slice
// at its window in both.
sdy.mesh @mesh = <["x"=2, "y"=2]>
func.func @two_dims(%a: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %b: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<16x16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %l = stablehlo.slice %b [1:14, 2:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>) -> tensor<13x13xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<13x13xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c1, %c2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>, tensor<13x13xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @two_dims
// CHECK: %[[N:.+]] = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<16x16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[N]] [1:14, 2:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>) -> tensor<13x13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg0, %[[W]], %{{.+}}, %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>, tensor<13x13xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
// CHECK-NEXT: return %[[D]]

// -----

// A sharded and an unsharded dimension narrowed: only the sharded one is
// widened; the slice along the unsharded one is local and stays.
sdy.mesh @mesh = <["x"=2]>
func.func @mixed_dims(%a: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %b: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %l = stablehlo.slice %b [1:7, 2:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>) -> tensor<6x13xf32>
  %u = stablehlo.negate %l {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<6x13xf32>
  %d = stablehlo.dynamic_update_slice %a, %u, %c1, %c2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>, tensor<6x13xf32>, tensor<i32>, tensor<i32>) -> tensor<8x16xf32>
  return %d : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @mixed_dims
// CHECK: %[[L:.+]] = stablehlo.slice %arg1 [1:7, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>) -> tensor<6x16xf32>
// CHECK-NEXT: %[[N:.+]] = stablehlo.negate %[[L]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<6x16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[N]] [0:6, 2:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<6x16xf32>) -> tensor<6x13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg0, %[[W]], %{{.+}}, %{{.+}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>, tensor<6x13xf32>, tensor<i32>, tensor<i32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[D]]

// -----

// A stencil on an argument of the DUS's own extent: widening its shifted
// reads back to that extent would step outside the argument. The rows that
// would come from outside are never read, so each read becomes a rotation,
// the cheapest halo exchange, and the DUS gets the slice at its window.
sdy.mesh @mesh = <["x"=2]>
func.func @rotate_argument(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %l = stablehlo.slice %a [0:13] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
  %r = stablehlo.slice %a [2:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
  %u = stablehlo.add %l, %r {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %b, %u, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @rotate_argument
// CHECK-DAG: %[[L:.+]] = "enzymexla.rotate"(%arg0) <{amount = 15 : i32, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[R:.+]] = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[U:.+]] = stablehlo.add %[[L]], %[[R]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[U]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg1, %[[W]]
// CHECK-NEXT: return %[[D]]

// -----

// The stencil's reads come as one lowered multislice (the form the regular
// invocations see): each result is a slice shifted by its index, and is
// widened like one.
sdy.mesh @mesh = <["x"=2]>
func.func @multislice_custom_call(%a: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, %b: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<16xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %ms:2 = stablehlo.custom_call @_SPMDInternalOp_MultiSlice(%a) {backend_config = "dimension=0,amount=1,start_indices=[1],limit_indices=[14],strides=[1],bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> (tensor<13xf32>, tensor<13xf32>)
  %u = stablehlo.add %ms#0, %ms#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
  %d = stablehlo.dynamic_update_slice %b, %u, %c1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>, tensor<13xf32>, tensor<i32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// CHECK-LABEL: func.func @multislice_custom_call
// CHECK: %[[R:.+]] = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<16xf32>
// CHECK-NEXT: %[[U:.+]] = stablehlo.add %arg0, %[[R]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<16xf32>
// CHECK-NEXT: %[[W:.+]] = stablehlo.slice %[[U]] [1:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<13xf32>
// CHECK-NEXT: %[[D:.+]] = stablehlo.dynamic_update_slice %arg1, %[[W]]
// CHECK-NEXT: return %[[D]]

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
// CHECK: %[[N:.+]] = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<13xf32>
// CHECK-NEXT: stablehlo.dynamic_update_slice %arg0, %[[N]]
