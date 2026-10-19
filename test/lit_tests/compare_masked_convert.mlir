// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=compare_convert" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --inline --canonicalize --symbol-dce --drop-unsupported-attributes | stablehlo-translate --interpret

// Equality of a masked extension only needs source-width bits when the mask
// and expected value both fit that width. The high-bit case covers negative
// i8 inputs; wider masks/expected values and ordered comparisons must not be
// blindly truncated. Keep the element order through the intervening reshape.

// CHECK-LABEL: func.func @low_bit(
// CHECK-NOT: stablehlo.convert
// CHECK: stablehlo.reshape {{.*}} -> tensor<10xi8>
// CHECK: stablehlo.and {{.*}} : tensor<10xi8>
// CHECK: stablehlo.compare EQ{{.*}}tensor<10xi8>
// CHECK-NOT: stablehlo.convert
// CHECK: return
func.func @low_bit(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<1> : tensor<10xi32>
  %expected = stablehlo.constant dense<0> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %flat, %mask : tensor<10xi32>
  %result = stablehlo.compare EQ, %bits, %expected, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

// CHECK-LABEL: func.func @high_bit(
// CHECK-NOT: stablehlo.convert
// CHECK: stablehlo.reshape {{.*}} -> tensor<10xi8>
// CHECK: stablehlo.and {{.*}} : tensor<10xi8>
// CHECK: stablehlo.compare EQ{{.*}}tensor<10xi8>
// CHECK-NOT: stablehlo.convert
// CHECK: return
func.func @high_bit(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<128> : tensor<10xi32>
  %expected = stablehlo.constant dense<128> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %flat, %mask : tensor<10xi32>
  %result = stablehlo.compare EQ, %bits, %expected, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

// CHECK-LABEL: func.func @not_equal(
// CHECK-NOT: stablehlo.convert
// CHECK: stablehlo.reshape {{.*}} -> tensor<10xi8>
// CHECK: stablehlo.and {{.*}} : tensor<10xi8>
// CHECK: stablehlo.compare NE{{.*}}tensor<10xi8>
// CHECK-NOT: stablehlo.convert
// CHECK: return
func.func @not_equal(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<3> : tensor<10xi32>
  %expected = stablehlo.constant dense<1> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %mask, %flat : tensor<10xi32>
  %result = stablehlo.compare NE, %expected, %bits, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

// CHECK-LABEL: func.func @wide_mask(
// CHECK: stablehlo.convert {{.*}} -> tensor<10x1xi32>
// CHECK: stablehlo.and {{.*}} : tensor<10xi32>
// CHECK: return
func.func @wide_mask(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<256> : tensor<10xi32>
  %expected = stablehlo.constant dense<256> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %flat, %mask : tensor<10xi32>
  %result = stablehlo.compare EQ, %bits, %expected, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

// CHECK-LABEL: func.func @wide_expected(
// CHECK: stablehlo.convert {{.*}} -> tensor<10x1xi32>
// CHECK: stablehlo.and {{.*}} : tensor<10xi32>
// CHECK: return
func.func @wide_expected(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<1> : tensor<10xi32>
  %expected = stablehlo.constant dense<256> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %flat, %mask : tensor<10xi32>
  %result = stablehlo.compare EQ, %bits, %expected, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

// CHECK-LABEL: func.func @ordered(
// CHECK: stablehlo.convert {{.*}} -> tensor<10x1xi32>
// CHECK: stablehlo.and {{.*}} : tensor<10xi32>
// CHECK: return
func.func @ordered(%x: tensor<10x1xi8>) -> tensor<10xi1> {
  %mask = stablehlo.constant dense<128> : tensor<10xi32>
  %expected = stablehlo.constant dense<1> : tensor<10xi32>
  %wide = stablehlo.convert %x : (tensor<10x1xi8>) -> tensor<10x1xi32>
  %flat = stablehlo.reshape %wide : (tensor<10x1xi32>) -> tensor<10xi32>
  %bits = stablehlo.and %flat, %mask : tensor<10xi32>
  %result = stablehlo.compare LT, %bits, %expected, SIGNED : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %result : tensor<10xi1>
}

func.func @main() {
  %x = stablehlo.constant dense<[[-128], [-127], [-3], [-2], [-1], [0], [1], [2], [3], [127]]> : tensor<10x1xi8>
  %e0 = stablehlo.constant dense<[true, false, false, true, false, true, false, true, false, false]> : tensor<10xi1>
  %r0 = func.call @low_bit(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r0, %e0 : tensor<10xi1>
  %e1 = stablehlo.constant dense<[true, true, true, true, true, false, false, false, false, false]> : tensor<10xi1>
  %r1 = func.call @high_bit(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r1, %e1 : tensor<10xi1>
  %e2 = stablehlo.constant dense<[true, false, false, true, true, true, false, true, true, true]> : tensor<10xi1>
  %r2 = func.call @not_equal(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r2, %e2 : tensor<10xi1>
  %e3 = stablehlo.constant dense<[true, true, true, true, true, false, false, false, false, false]> : tensor<10xi1>
  %r3 = func.call @wide_mask(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r3, %e3 : tensor<10xi1>
  %e4 = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false]> : tensor<10xi1>
  %r4 = func.call @wide_expected(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r4, %e4 : tensor<10xi1>
  %e5 = stablehlo.constant dense<[false, false, false, false, false, true, true, true, true, true]> : tensor<10xi1>
  %r5 = func.call @ordered(%x) : (tensor<10x1xi8>) -> tensor<10xi1>
  check.expect_eq %r5, %e5 : tensor<10xi1>
  return
}
