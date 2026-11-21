// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=short_row_reduce_pad" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A row of 27: padded with zeros to 32.
func.func @row27(%x: tensor<64x27xf64>) -> tensor<64xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %r = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x27xf64>, tensor<f64>) -> tensor<64xf64>
  return %r : tensor<64xf64>
}

// CHECK:  func.func @row27(%arg0: tensor<64x27xf64>) -> tensor<64xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0], high = [0, 5], interior = [0, 0] : (tensor<64x27xf64>, tensor<f64>) -> tensor<64x32xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:    return %1 : tensor<64xf64>
// CHECK-NEXT:  }

// -----

// Two trailing dimensions making a row of 27, a maximum with an init that is
// not its identity: the rows are padded with -inf, the init kept.
func.func @rows3x9(%x: tensor<8x4x3x9xf32>) -> tensor<8x4xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %r = stablehlo.reduce(%x init: %cst) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<8x4x3x9xf32>, tensor<f32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// CHECK:  func.func @rows3x9(%arg0: tensor<8x4x3x9xf32>) -> tensor<8x4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<8x4x3x9xf32>) -> tensor<8x4x27xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst_0, low = [0, 0, 0], high = [0, 0, 5], interior = [0, 0, 0] : (tensor<8x4x27xf32>, tensor<f32>) -> tensor<8x4x32xf32>
// CHECK-NEXT:    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x32xf32>, tensor<f32>) -> tensor<8x4xf32>
// CHECK-NEXT:    return %2 : tensor<8x4xf32>
// CHECK-NEXT:  }

// -----

// Rows of 16 and 32 are left to XLA's reduction emitter, a leading dimension
// to its column reduction.
func.func @unchanged(%a: tensor<64x16xf64>, %b: tensor<64x32xf64>, %c: tensor<27x64xf64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %ra = stablehlo.reduce(%a init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16xf64>, tensor<f64>) -> tensor<64xf64>
  %rb = stablehlo.reduce(%b init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf64>, tensor<f64>) -> tensor<64xf64>
  %rc = stablehlo.reduce(%c init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<27x64xf64>, tensor<f64>) -> tensor<64xf64>
  return %ra, %rb, %rc : tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
}

// CHECK:  func.func @unchanged(%arg0: tensor<64x16xf64>, %arg1: tensor<64x32xf64>, %arg2: tensor<27x64xf64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:    %1 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<27x64xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:    return %0, %1, %2 : tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
// CHECK-NEXT:  }
