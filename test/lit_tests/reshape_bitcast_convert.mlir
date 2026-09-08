// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reshape_bitcast_convert" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The bitcast adds a trailing dimension of four bytes per f32. Folding the
// two surrounding reshapes retains that dimension and the bitcast attributes.
// CHECK-LABEL: func.func @narrow(
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert {{.*}} {mhlo.sharding = "{replicated}"} : (tensor<2x3xf32>) -> tensor<2x3x4xi8>
// CHECK-NOT: stablehlo.reshape
// CHECK: return
func.func @narrow(%x: tensor<2x3xf32>) -> tensor<2x3x4xi8> {
  %flat = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
  %bytes = stablehlo.bitcast_convert %flat {mhlo.sharding = "{replicated}"} : (tensor<6xf32>) -> tensor<?x4xi8>
  %r = stablehlo.reshape %bytes : (tensor<?x4xi8>) -> tensor<2x3x4xi8>
  return %r : tensor<2x3x4xi8>
}

// Widening removes the same innermost group. Same-width bitcasts preserve
// the entire shape. Neither case needs intermediate flat views.
// CHECK-LABEL: func.func @widen(
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<2x3x4xi8>) -> tensor<2x3xf32>
// CHECK-NOT: stablehlo.reshape
// CHECK: return
func.func @widen(%x: tensor<2x3x4xi8>) -> tensor<2x3xf32> {
  %flat = stablehlo.reshape %x : (tensor<2x3x4xi8>) -> tensor<6x4xi8>
  %words = stablehlo.bitcast_convert %flat : (tensor<6x4xi8>) -> tensor<6xf32>
  %r = stablehlo.reshape %words : (tensor<6xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @same_width(
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<2x3xf32>) -> tensor<2x3xi32>
// CHECK-NOT: stablehlo.reshape
// CHECK: return
func.func @same_width(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %flat = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
  %words = stablehlo.bitcast_convert %flat : (tensor<6xf32>) -> tensor<6xi32>
  %r = stablehlo.reshape %words : (tensor<6xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// A 3x8 result mixes the four-byte groups with other dimensions. A direct
// bitcast of a 2x3 f32 tensor cannot have this shape, so keep both reshapes.
// CHECK-LABEL: func.func @different_grouping(
// CHECK: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert
// CHECK: stablehlo.reshape {{.*}} -> tensor<3x8xi8>
func.func @different_grouping(%x: tensor<2x3xf32>) -> tensor<3x8xi8> {
  %flat = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
  %bytes = stablehlo.bitcast_convert %flat : (tensor<6xf32>) -> tensor<6x4xi8>
  %r = stablehlo.reshape %bytes : (tensor<6x4xi8>) -> tensor<3x8xi8>
  return %r : tensor<3x8xi8>
}

// Do not duplicate a bitcast with another live user.
// CHECK-LABEL: func.func @shared_bitcast(
// CHECK: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert
// CHECK: stablehlo.reshape
// CHECK: return
func.func @shared_bitcast(%x: tensor<2x3xf32>) -> (tensor<2x3x4xi8>, tensor<6x4xi8>) {
  %flat = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
  %bytes = stablehlo.bitcast_convert %flat : (tensor<6xf32>) -> tensor<6x4xi8>
  %r = stablehlo.reshape %bytes : (tensor<6x4xi8>) -> tensor<2x3x4xi8>
  return %r, %bytes : tensor<2x3x4xi8>, tensor<6x4xi8>
}
