// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @identity(
func.func @identity(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.identity %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.identity %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @exponent(
func.func @exponent(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.exponent %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.exponent %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @exponent_base2(
func.func @exponent_base2(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.exponent_base2 %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.exponent_base2 %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @exponent_base10(
func.func @exponent_base10(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.exponent_base10 %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.exponent_base10 %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @logarithm(
func.func @logarithm(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.logarithm %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.logarithm %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @logarithm_base2(
func.func @logarithm_base2(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.logarithm_base2 %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.logarithm_base2 %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @logarithm_base10(
func.func @logarithm_base10(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.logarithm_base10 %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.logarithm_base10 %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @square(
func.func @square(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.square %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.square %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @square_root(
func.func @square_root(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.square_root %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.square_root %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reciprocal_square_root(
func.func @reciprocal_square_root(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.reciprocal_square_root %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.reciprocal_square_root %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reciprocal(
func.func @reciprocal(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.reciprocal %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.reciprocal %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @negative(
func.func @negative(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.negative %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.negative %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sign(
func.func @sign(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.sign %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.sign %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @ceil(
func.func @ceil(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.ceil %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.ceil %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @floor(
func.func @floor(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.floor %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.floor %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @round(
func.func @round(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.round %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.round %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @rint(
func.func @rint(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.rint %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.rint %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @truncate(
func.func @truncate(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.truncate %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.truncate %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sin(
func.func @sin(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.sin %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.sin %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @cos(
func.func @cos(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.cos %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.cos %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @tan(
func.func @tan(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.tan %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.tan %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sinh(
func.func @sinh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.sinh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.sinh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @cosh(
func.func @cosh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.cosh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.cosh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @tanh(
func.func @tanh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.tanh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.tanh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @asin(
func.func @asin(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.asin %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.asin %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @acos(
func.func @acos(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.acos %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.acos %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @atan(
func.func @atan(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.atan %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.atan %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @asinh(
func.func @asinh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.asinh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.asinh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @acosh(
func.func @acosh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.acosh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.acosh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @atanh(
func.func @atanh(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.atanh %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.atanh %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @erf(
func.func @erf(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.erf %{{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.erf %x : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @not(
func.func @not(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.not %{{.+}} : (tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.not %b : (tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @conjugate(
func.func @conjugate(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.conjugate %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  %0 = mps.conjugate %c : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_not(
func.func @bitwise_not(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_not %{{.+}} : (tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_not %i : (tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @absolute(
func.func @absolute(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.absolute %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %0 = mps.absolute %c : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @absolute_square(
func.func @absolute_square(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.absolute_square %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %0 = mps.absolute_square %c : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @real_part(
func.func @real_part(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.real_part %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %0 = mps.real_part %c : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @imaginary_part(
func.func @imaginary_part(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.imaginary_part %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %0 = mps.imaginary_part %c : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_population_count(
func.func @bitwise_population_count(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_population_count %{{.+}} : (tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_population_count %i : (tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @signbit(
func.func @signbit(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.signbit %{{.+}} : (tensor<4xf32>) -> tensor<4xi1>
  %0 = mps.signbit %x : (tensor<4xf32>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @is_infinite(
func.func @is_infinite(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.is_infinite %{{.+}} : (tensor<4xf32>) -> tensor<4xi1>
  %0 = mps.is_infinite %x : (tensor<4xf32>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @is_finite(
func.func @is_finite(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.is_finite %{{.+}} : (tensor<4xf32>) -> tensor<4xi1>
  %0 = mps.is_finite %x : (tensor<4xf32>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @is_nan(
func.func @is_nan(%x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.is_nan %{{.+}} : (tensor<4xf32>) -> tensor<4xi1>
  %0 = mps.is_nan %x : (tensor<4xf32>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @planar(
func.func @planar(%c: tensor<4xcomplex<f32>>) {
  // CHECK: %{{.+}} = mps.planar %{{.+}} : (tensor<4xcomplex<f32>>) -> tensor<2x4xf32>
  %0 = mps.planar %c : (tensor<4xcomplex<f32>>) -> tensor<2x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @addition(
func.func @addition(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.addition %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.addition %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @subtraction(
func.func @subtraction(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.subtraction %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.subtraction %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @multiplication(
func.func @multiplication(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.multiplication %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.multiplication %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @division(
func.func @division(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.division %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.division %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @division_no_nan(
func.func @division_no_nan(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.division_no_nan %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.division_no_nan %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @modulo(
func.func @modulo(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.modulo %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.modulo %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @floor_modulo(
func.func @floor_modulo(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.floor_modulo %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.floor_modulo %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @power(
func.func @power(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.power %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.power %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @minimum(
func.func @minimum(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.minimum %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.minimum %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @maximum(
func.func @maximum(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.maximum %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.maximum %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @minimum_with_nan_propagation(
func.func @minimum_with_nan_propagation(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.minimum_with_nan_propagation %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.minimum_with_nan_propagation %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @maximum_with_nan_propagation(
func.func @maximum_with_nan_propagation(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.maximum_with_nan_propagation %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.maximum_with_nan_propagation %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @atan2(
func.func @atan2(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.atan2 %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %0 = mps.atan2 %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_and(
func.func @bitwise_and(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_and %{{.+}}, %{{.+}} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_and %i, %i : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_or(
func.func @bitwise_or(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_or %{{.+}}, %{{.+}} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_or %i, %i : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_xor(
func.func @bitwise_xor(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_xor %{{.+}}, %{{.+}} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_xor %i, %i : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_left_shift(
func.func @bitwise_left_shift(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_left_shift %{{.+}}, %{{.+}} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_left_shift %i, %i : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @bitwise_right_shift(
func.func @bitwise_right_shift(%i: tensor<4xi32>) {
  // CHECK: %{{.+}} = mps.bitwise_right_shift %{{.+}}, %{{.+}} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = mps.bitwise_right_shift %i, %i : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: func.func @logical_and(
func.func @logical_and(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_and %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_and %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @logical_or(
func.func @logical_or(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_or %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_or %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @logical_nand(
func.func @logical_nand(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_nand %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_nand %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @logical_nor(
func.func @logical_nor(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_nor %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_nor %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @logical_xor(
func.func @logical_xor(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_xor %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_xor %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @logical_xnor(
func.func @logical_xnor(%b: tensor<4xi1>) {
  // CHECK: %{{.+}} = mps.logical_xnor %{{.+}}, %{{.+}} : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  %0 = mps.logical_xnor %b, %b : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return
}

// -----

// CHECK-LABEL: func.func @equal(
func.func @equal(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.equal %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.equal %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @not_equal(
func.func @not_equal(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.not_equal %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.not_equal %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @less_than(
func.func @less_than(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.less_than %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.less_than %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @less_than_or_equal_to(
func.func @less_than_or_equal_to(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.less_than_or_equal_to %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.less_than_or_equal_to %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @greater_than(
func.func @greater_than(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.greater_than %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.greater_than %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @greater_than_or_equal_to(
func.func @greater_than_or_equal_to(%x: tensor<2x3xf32>, %y: tensor<3xf32>) {
  // CHECK: %{{.+}} = mps.greater_than_or_equal_to %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  %0 = mps.greater_than_or_equal_to %x, %y : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xi1>
  return
}

// -----

// CHECK-LABEL: func.func @complex(
func.func @complex(%x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.complex %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xcomplex<f32>>
  %0 = mps.complex %x, %x : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func.func @select(
func.func @select(%p: tensor<2x3xi1>, %x: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = mps.select %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = mps.select %p, %x, %x : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @clamp(
func.func @clamp(%x: tensor<2x3xf32>, %lo: tensor<f32>, %hi: tensor<f32>) {
  // CHECK: %{{.+}} = mps.clamp %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %0 = mps.clamp %x, %lo, %hi : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  return
}
