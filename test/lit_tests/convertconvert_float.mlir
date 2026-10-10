// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s --verify-each | FileCheck %s

// Lossy first conversions must survive, even when the final type is the source.
// CHECK-LABEL: func.func @f64_f32_f64
// CHECK: %[[F32:.*]] = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xf32>
// CHECK-NEXT: %[[F64:.*]] = stablehlo.convert %[[F32]] : (tensor<4xf32>) -> tensor<4xf64>
// CHECK-NEXT: return %[[F64]]
func.func @f64_f32_f64(%a: tensor<4xf64>) -> tensor<4xf64> {
  %mid = stablehlo.convert %a : (tensor<4xf64>) -> tensor<4xf32>
  %result = stablehlo.convert %mid : (tensor<4xf32>) -> tensor<4xf64>
  return %result : tensor<4xf64>
}

// CHECK-LABEL: func.func @f32_f16_f32
// CHECK: %[[F16:.*]] = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf16>
// CHECK-NEXT: %[[BACK32:.*]] = stablehlo.convert %[[F16]] : (tensor<4xf16>) -> tensor<4xf32>
// CHECK-NEXT: return %[[BACK32]]
func.func @f32_f16_f32(%a: tensor<4xf32>) -> tensor<4xf32> {
  %mid = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xf16>
  %result = stablehlo.convert %mid : (tensor<4xf16>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// Equal storage widths do not imply equal precision or exponent range.
// CHECK-LABEL: func.func @f16_bf16_f16
// CHECK: %[[BF:.*]] = stablehlo.convert %arg0 : (tensor<4xf16>) -> tensor<4xbf16>
// CHECK-NEXT: %[[BACK16:.*]] = stablehlo.convert %[[BF]] : (tensor<4xbf16>) -> tensor<4xf16>
// CHECK-NEXT: return %[[BACK16]]
func.func @f16_bf16_f16(%a: tensor<4xf16>) -> tensor<4xf16> {
  %mid = stablehlo.convert %a : (tensor<4xf16>) -> tensor<4xbf16>
  %result = stablehlo.convert %mid : (tensor<4xbf16>) -> tensor<4xf16>
  return %result : tensor<4xf16>
}

// CHECK-LABEL: func.func @bf16_f16_bf16
// CHECK: %[[HALF:.*]] = stablehlo.convert %arg0 : (tensor<4xbf16>) -> tensor<4xf16>
// CHECK-NEXT: %[[BACKBF:.*]] = stablehlo.convert %[[HALF]] : (tensor<4xf16>) -> tensor<4xbf16>
// CHECK-NEXT: return %[[BACKBF]]
func.func @bf16_f16_bf16(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %mid = stablehlo.convert %a : (tensor<4xbf16>) -> tensor<4xf16>
  %result = stablehlo.convert %mid : (tensor<4xf16>) -> tensor<4xbf16>
  return %result : tensor<4xbf16>
}

// A different final type must not collapse a lossy first conversion either.
// For example, 1 + 2^-11 + 2^-30 double-rounds through f32 on its way to f16.
// CHECK-LABEL: func.func @f64_f32_f16
// CHECK: %[[MID32:.*]] = stablehlo.convert %arg0 : (tensor<4xf64>) -> tensor<4xf32>
// CHECK-NEXT: %[[FINAL16:.*]] = stablehlo.convert %[[MID32]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK-NEXT: return %[[FINAL16]]
func.func @f64_f32_f16(%a: tensor<4xf64>) -> tensor<4xf16> {
  %mid = stablehlo.convert %a : (tensor<4xf64>) -> tensor<4xf32>
  %result = stablehlo.convert %mid : (tensor<4xf32>) -> tensor<4xf16>
  return %result : tensor<4xf16>
}

// Exactly widening the first conversion still permits simplification.
// CHECK-LABEL: func.func @f16_f32_f16
// CHECK-NEXT: return %arg0 : tensor<4xf16>
func.func @f16_f32_f16(%a: tensor<4xf16>) -> tensor<4xf16> {
  %mid = stablehlo.convert %a : (tensor<4xf16>) -> tensor<4xf32>
  %result = stablehlo.convert %mid : (tensor<4xf32>) -> tensor<4xf16>
  return %result : tensor<4xf16>
}

// CHECK-LABEL: func.func @bf16_f32_bf16
// CHECK-NEXT: return %arg0 : tensor<4xbf16>
func.func @bf16_f32_bf16(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %mid = stablehlo.convert %a : (tensor<4xbf16>) -> tensor<4xf32>
  %result = stablehlo.convert %mid : (tensor<4xf32>) -> tensor<4xbf16>
  return %result : tensor<4xbf16>
}

// CHECK-LABEL: func.func @f32_f64_f32
// CHECK-NEXT: return %arg0 : tensor<4xf32>
func.func @f32_f64_f32(%a: tensor<4xf32>) -> tensor<4xf32> {
  %mid = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xf64>
  %result = stablehlo.convert %mid : (tensor<4xf64>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @f16_f32_f64
// CHECK-NEXT: %[[DIRECT:.*]] = stablehlo.convert %arg0 : (tensor<4xf16>) -> tensor<4xf64>
// CHECK-NEXT: return %[[DIRECT]]
func.func @f16_f32_f64(%a: tensor<4xf16>) -> tensor<4xf64> {
  %mid = stablehlo.convert %a : (tensor<4xf16>) -> tensor<4xf32>
  %result = stablehlo.convert %mid : (tensor<4xf32>) -> tensor<4xf64>
  return %result : tensor<4xf64>
}
