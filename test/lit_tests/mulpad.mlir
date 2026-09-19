// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=mul_zero_pad<1>(1);negative_pad_to_slice<16>;" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefixes=CHECK,NONAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=mul_zero_pad<1>(0);negative_pad_to_slice<16>;" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefixes=CHECK,NAN

// mul_zero_pad rewrites pad(x, 0) * y into pad(x * slice(y), 0), which forces the
// newly-padded region to 0 rather than to 0 * y. That is only sound when y cannot be
// NaN or +-Inf there, so the pattern is gated on its NoNan parameter.
//
// This runs mul_zero_pad on its own rather than the whole --enzyme-hlo-opt pipeline:
// that pipeline also contains binop_pad_to_concat_mul, which performs the same
// rewrite ungated and would hide the gating being tested here.

// CHECK-LABEL: func.func @pad_multiply(
// NONAN:         %[[SLICE:.+]] = stablehlo.slice %arg1 [0:1, 0:3, 1024:2048]
// NONAN:         %[[MUL:.+]] = stablehlo.multiply %arg0, %[[SLICE]] : tensor<1x3x1024xf32>
// NONAN:         stablehlo.pad %[[MUL]], %{{.*}}, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0]
// NAN-NOT:       stablehlo.slice
// NAN:           %[[PAD:.+]] = stablehlo.pad %arg0, %{{.*}}, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0]
// NAN:           stablehlo.multiply %[[PAD]], %arg1
func.func @pad_multiply(%4: tensor<1x3x1024xf32>, %2: tensor<1x3x2048xf32>) -> tensor<1x3x2048xf32> {
  %constant_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.pad %4, %constant_0, low = [0, 0, 1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x1024xf32>, tensor<f32>) -> tensor<1x3x2048xf32>
  %7 = stablehlo.multiply %5, %2 : tensor<1x3x2048xf32>
  return %7 : tensor<1x3x2048xf32>
}

// The pad here has a negative low padding, so mul_zero_pad never applies and
// negative_pad_to_slice turns it into a slice regardless of the NoNan parameter.
func.func @pad_multiply_inv(%4: tensor<1x3x2048xf32>, %2: tensor<1x3x1024xf32>) -> tensor<1x3x1024xf32> {
  %constant_0 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.pad %4, %constant_0, low = [0, 0, -1024], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x3x2048xf32>, tensor<f32>) -> tensor<1x3x1024xf32>
  %7 = stablehlo.multiply %5, %2 : tensor<1x3x1024xf32>
  return %7 : tensor<1x3x1024xf32>
}

// CHECK-LABEL: func.func @pad_multiply_inv(
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.slice %arg0 [0:1, 0:3, 1024:2048] : (tensor<1x3x2048xf32>) -> tensor<1x3x1024xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.multiply %[[i0]], %arg1 : tensor<1x3x1024xf32>
// CHECK-NEXT:    return %[[i1]] : tensor<1x3x1024xf32>
// CHECK-NEXT:  }
