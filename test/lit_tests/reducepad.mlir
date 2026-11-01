// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=REDUCE_PAD

module {
  func.func @main(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
    %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
    %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
    return %conv : tensor<6x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3x1xf32>, %arg1: tensor<f32>) -> tensor<6x1xf32> {
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %arg1, low = [1, 0], high = [3, 0], interior = [0, 0] : (tensor<2x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<6x1xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
    %pv = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
    %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
    return %conv : tensor<6x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x3x1xf32>, %arg1: tensor<f32>) -> tensor<6x1xf32> {
// CHECK-DAG: %[[ADDED:.+]] = stablehlo.constant dense<2.400000e+01> : tensor<2x1xf32>
// CHECK-DAG: %[[FULL:.+]] = stablehlo.constant dense<3.300000e+01> : tensor<f32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x1xf32>
// CHECK: %[[SUM:.+]] = stablehlo.add %[[RED]], %[[ADDED]] : tensor<2x1xf32>
// CHECK: %[[FILL:.+]] = stablehlo.add %arg1, %[[FULL]] : tensor<f32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[SUM]], %[[FILL]], low = [1, 0], high = [3, 0], interior = [0, 0] : (tensor<2x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK: return %[[PAD]] : tensor<6x1xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[BROADCAST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[FOLD:.+]] = stablehlo.maximum %[[RED]], %[[BROADCAST]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[FOLD]], %[[ZERO]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.minimum across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[BROADCAST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.minimum across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[FOLD:.+]] = stablehlo.minimum %[[RED]], %[[BROADCAST]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[FOLD]], %[[ZERO]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst, low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK-NEXT:     return %2 : tensor<27xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.100000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ADDED:.+]] = stablehlo.constant dense<15.8631029> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[FULL:.+]] = stablehlo.constant dense<{{(21[.]113[0-9]+|2[.]1113[0-9]+e\+01)}}> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.multiply across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[PROD:.+]] = stablehlo.multiply %[[RED]], %[[ADDED]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[PROD]], %[[FULL]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }


// A constant adjoint pads a row before its lanes are summed. The row's
// contribution is -3, rather than a single -1.
func.func @sum_padded_row(%arg: tensor<1x3xf64>, %init: tensor<f64>) -> tensor<2xf64> {
  %pv = stablehlo.constant dense<-1.0> : tensor<f64>
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// REDUCE_PAD-LABEL: func.func @sum_padded_row(
// REDUCE_PAD-DAG: %[[PV:.*]] = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
// REDUCE_PAD-DAG: %[[N:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD: %[[M:.*]] = stablehlo.multiply %[[PV]], %[[N]] : tensor<f64>
// REDUCE_PAD: %[[F:.*]] = stablehlo.add %arg1, %[[M]] : tensor<f64>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// With several reduced axes, the number of padded elements is the difference
// between slice volumes, not the sum of padding widths: 6*5 - 2*3 = 24.
func.func @sum_multiple_axes(%arg: tensor<2x2x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<3xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 1, 1], high = [1, 3, 1], interior = [0, 0, 0] : (tensor<2x2x3xf64>, tensor<f64>) -> tensor<3x6x5xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1, 2] : (tensor<3x6x5xf64>, tensor<f64>) -> tensor<3xf64>
  return %r : tensor<3xf64>
}
// REDUCE_PAD-LABEL: func.func @sum_multiple_axes(
// REDUCE_PAD-DAG: %[[N:.*]] = stablehlo.constant dense<2.400000e+01> : tensor<f64>
// REDUCE_PAD-DAG: %[[TOTAL:.*]] = stablehlo.constant dense<3.000000e+01> : tensor<f64>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.add across dimensions = [1, 2]
// REDUCE_PAD: %[[M:.*]] = stablehlo.multiply %arg1, %[[N]] : tensor<f64>
// REDUCE_PAD: %[[B:.*]] = stablehlo.broadcast_in_dim %[[M]], dims = []
// REDUCE_PAD: %[[A:.*]] = stablehlo.add %[[R]], %[[B]]
// REDUCE_PAD: %[[FULL:.*]] = stablehlo.multiply %arg1, %[[TOTAL]] : tensor<f64>
// REDUCE_PAD: %[[F:.*]] = stablehlo.add %arg2, %[[FULL]] : tensor<f64>
// REDUCE_PAD: stablehlo.pad %[[A]], %[[F]], low = [0], high = [1], interior = [0]

func.func @product_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// REDUCE_PAD-LABEL: func.func @product_padded_row(
// REDUCE_PAD-DAG: %[[N:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.multiply across dimensions = [1]
// REDUCE_PAD: %[[P:.*]] = stablehlo.power %arg1, %[[N]] : tensor<f64>
// REDUCE_PAD: %[[F:.*]] = stablehlo.multiply %arg2, %[[P]] : tensor<f64>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// No padding along the reduced axes means the fill never participates in
// the original row's minimum or maximum. It still combines with init in a
// padded output row.
func.func @minimum_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// REDUCE_PAD-LABEL: func.func @minimum_padded_row(

// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.minimum across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.minimum %[[R]]
// REDUCE_PAD: %[[F:.*]] = stablehlo.minimum %arg2, %arg1 : tensor<f64>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

func.func @maximum_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// REDUCE_PAD-LABEL: func.func @maximum_padded_row(

// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.maximum across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.maximum %[[R]]
// REDUCE_PAD: %[[F:.*]] = stablehlo.maximum %arg2, %arg1 : tensor<f64>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// Interior padding contributes both reduced elements and output lanes.
func.func @sum_interior(%arg: tensor<2x3xi32>, %pv: tensor<i32>, %init: tensor<i32>) -> tensor<5xi32> {
  %p = stablehlo.pad %arg, %pv, low = [1, 0], high = [1, 0], interior = [1, 1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x5xi32>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<5x5xi32>, tensor<i32>) -> tensor<5xi32>
  return %r : tensor<5xi32>
}
// REDUCE_PAD-LABEL: func.func @sum_interior(
// REDUCE_PAD-DAG: %[[N:.*]] = stablehlo.constant dense<2> : tensor<i32>
// REDUCE_PAD-DAG: %[[TOTAL:.*]] = stablehlo.constant dense<5> : tensor<i32>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD: %[[M:.*]] = stablehlo.multiply %arg1, %[[N]] : tensor<i32>
// REDUCE_PAD: %[[B:.*]] = stablehlo.broadcast_in_dim %[[M]], dims = []
// REDUCE_PAD: %[[A:.*]] = stablehlo.add %[[R]], %[[B]]
// REDUCE_PAD: %[[FULL:.*]] = stablehlo.multiply %arg1, %[[TOTAL]] : tensor<i32>
// REDUCE_PAD: %[[F:.*]] = stablehlo.add %arg2, %[[FULL]] : tensor<i32>
// REDUCE_PAD: stablehlo.pad %[[A]], %[[F]], low = [1], high = [1], interior = [1]

// An empty reduced axis has no fill elements: padded rows return init too.
func.func @empty_axis(%arg: tensor<1x0xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x0xf64>, tensor<f64>) -> tensor<2x0xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<2x0xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// REDUCE_PAD-LABEL: func.func @empty_axis(

// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.minimum across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.minimum %[[R]]
// REDUCE_PAD: stablehlo.pad %[[R]], %arg2, low = [0], high = [1], interior = [0]

// A product exponent must fit the element type. Do not turn 128 copies of
// an i8 value into a power with a wrapped, negative exponent.
func.func @product_exponent_overflow(%arg: tensor<1x128xi8>, %pv: tensor<i8>, %init: tensor<i8>) -> tensor<2xi8> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xi8>, tensor<i8>) -> tensor<2x128xi8>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x128xi8>, tensor<i8>) -> tensor<2xi8>
  return %r : tensor<2xi8>
}
// REDUCE_PAD-LABEL: func.func @product_exponent_overflow(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.multiply across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.power
// REDUCE_PAD: return %[[R]]

// In f16, 2049 rounds to 2048. A product of 2049 copies of -1 is -1,
// whereas power(-1, 2048) is +1. Retain the reduction when the exponent
// cannot be represented exactly, including a complex element's real type.
func.func @product_float_exponent_rounding(%arg: tensor<1x2049xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf16> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2049xf16>, tensor<f16>) -> tensor<2x2049xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x2049xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @product_float_exponent_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.multiply across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.power
// REDUCE_PAD: return %[[R]]

// The full slice volume 2050 is exact in f16, but the 2049 missing elements
// are not. Check the correction exponent as well as the output fill.
func.func @product_missing_exponent_rounding(%arg: tensor<1x1xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf16> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 2049], interior = [0, 0] : (tensor<1x1xf16>, tensor<f16>) -> tensor<2x2050xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x2050xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @product_missing_exponent_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.multiply across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.power
// REDUCE_PAD: return %[[R]]

func.func @product_complex_exponent_rounding(%arg: tensor<1x16777217xcomplex<f32>>, %pv: tensor<complex<f32>>, %init: tensor<complex<f32>>) -> tensor<2xcomplex<f32>> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x16777217xcomplex<f32>>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
  return %r : tensor<2xcomplex<f32>>
}
// REDUCE_PAD-LABEL: func.func @product_complex_exponent_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.multiply across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.power
// REDUCE_PAD: return %[[R]]

// A zero additive fill needs no multiplier, even when the count overflows f16.
func.func @sum_zero_factor_overflow(%arg: tensor<1x65536xf16>) -> tensor<2xf16> {
  %pv = stablehlo.constant dense<0.0> : tensor<f16>
  %init = stablehlo.constant dense<7.0> : tensor<f16>
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x65536xf16>, tensor<f16>) -> tensor<2x65536xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x65536xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @sum_zero_factor_overflow(
// REDUCE_PAD-DAG: %[[INIT:.*]] = stablehlo.constant dense<7.000000e+00> : tensor<f16>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %[[INIT]]) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: stablehlo.pad %[[R]], %[[INIT]], low = [0], high = [1], interior = [0]

// Unknown padding may be tiny and have a finite sum, even though count is Inf.
func.func @sum_float_factor_overflow(%arg: tensor<1x65536xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf16> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x65536xf16>, tensor<f16>) -> tensor<2x65536xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x65536xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @sum_float_factor_overflow(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: return %[[R]]

func.func @sum_float_factor_rounding(%arg: tensor<1x2049xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf16> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2049xf16>, tensor<f16>) -> tensor<2x2049xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x2049xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @sum_float_factor_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: return %[[R]]

// The full count 2050 is exact, but the correction count 2049 is not.
func.func @sum_missing_factor_rounding(%arg: tensor<1x1xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf16> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 2049], interior = [0, 0] : (tensor<1x1xf16>, tensor<f16>) -> tensor<2x2050xf16>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x2050xf16>, tensor<f16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// REDUCE_PAD-LABEL: func.func @sum_missing_factor_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: return %[[R]]

func.func @sum_complex_factor_rounding(%arg: tensor<1x16777217xcomplex<f32>>, %pv: tensor<complex<f32>>, %init: tensor<complex<f32>>) -> tensor<2xcomplex<f32>> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x16777217xcomplex<f32>>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
  return %r : tensor<2xcomplex<f32>>
}
// REDUCE_PAD-LABEL: func.func @sum_complex_factor_rounding(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: return %[[R]]

func.func @sum_complex_zero_large_count(%arg: tensor<1x16777217xcomplex<f32>>, %init: tensor<complex<f32>>) -> tensor<2xcomplex<f32>> {
  %pv = stablehlo.constant dense<(0.0, 0.0)> : tensor<complex<f32>>
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x16777217xcomplex<f32>>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x16777217xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
  return %r : tensor<2xcomplex<f32>>
}
// REDUCE_PAD-LABEL: func.func @sum_complex_zero_large_count(
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: stablehlo.pad %[[R]], %arg1, low = [0], high = [1], interior = [0]

// Boolean Add is OR: an even number of true values still contributes true.
func.func @sum_boolean_even_count(%arg: tensor<1x2xi1>, %init: tensor<i1>) -> tensor<2xi1> {
  %pv = stablehlo.constant dense<true> : tensor<i1>
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2xi1>, tensor<i1>) -> tensor<2x2xi1>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x2xi1>, tensor<i1>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// REDUCE_PAD-LABEL: func.func @sum_boolean_even_count(
// REDUCE_PAD-DAG: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<i1>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD-NOT: stablehlo.multiply
// REDUCE_PAD: %[[F:.*]] = stablehlo.add %arg1, %[[TRUE]] : tensor<i1>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// A promoted reducer is valid, but the default ReduceOp builder infers f16.
func.func @sum_promoted_reducer(%arg: tensor<1x3xf16>, %pv: tensor<f16>, %init: tensor<f16>) -> tensor<2xf32> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf16>, tensor<f16>) -> tensor<2x3xf16>
  %r = "stablehlo.reduce"(%p, %init) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %s = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<2x3xf16>, tensor<f16>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// REDUCE_PAD-LABEL: func.func @sum_promoted_reducer(
// REDUCE_PAD: %[[P:.*]] = stablehlo.pad %arg0, %arg1
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%[[P]] init: %arg2) across dimensions = [1]
// REDUCE_PAD: stablehlo.add {{.*}} : tensor<f32>
// REDUCE_PAD: return %[[R]] : tensor<2xf32>

// Scale complex components with a real count: (Inf, 1) + (Inf, 1) = (Inf, 2).
// A complex multiplication by (2, 0) would introduce Inf*0 into the imaginary part.
func.func @sum_complex_components(%arg: tensor<1x2xcomplex<f32>>, %pv: tensor<complex<f32>>, %init: tensor<complex<f32>>) -> tensor<2xcomplex<f32>> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x2xcomplex<f32>>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
  return %r : tensor<2xcomplex<f32>>
}
// REDUCE_PAD-LABEL: func.func @sum_complex_components(
// REDUCE_PAD-DAG: %[[N:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// REDUCE_PAD: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.add across dimensions = [1]
// REDUCE_PAD: %[[REAL:.*]] = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
// REDUCE_PAD: %[[IMAG:.*]] = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
// REDUCE_PAD: %[[RS:.*]] = stablehlo.multiply %[[REAL]], %[[N]] : tensor<f32>
// REDUCE_PAD: %[[IS:.*]] = stablehlo.multiply %[[IMAG]], %[[N]] : tensor<f32>
// REDUCE_PAD: %[[C:.*]] = stablehlo.complex %[[RS]], %[[IS]] : tensor<complex<f32>>
// REDUCE_PAD: %[[F:.*]] = stablehlo.add %arg2, %[[C]] : tensor<complex<f32>>
// REDUCE_PAD: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]
