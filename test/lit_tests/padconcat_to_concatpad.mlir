// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=pad_concat_to_concat_pad},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

func.func @test_pad_leftover(%arg0 : tensor<128x2031x2032xf64>, %arg1 : tensor<1x2032x2032xf64>, %arg2: tensor<1x2032x2032xf64>) -> tensor<130x2033x2032xf64> {
  %cst_29 = stablehlo.constant dense<0.5> : tensor<f64>
  %p1 = stablehlo.pad %arg0, %cst_29, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<128x2031x2032xf64>, tensor<f64>) -> tensor<128x2033x2032xf64>
  %p2 = stablehlo.pad %arg1, %cst_29, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2032x2032xf64>, tensor<f64>) -> tensor<1x2033x2032xf64> 
  %p3 = stablehlo.pad %arg2, %cst_29, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2032x2032xf64>, tensor<f64>) -> tensor<1x2033x2032xf64> 

  %concat = stablehlo.concatenate %p2,%p1,%p3, dim = 0 : (tensor<1x2033x2032xf64>, tensor<128x2033x2032xf64>, tensor<1x2033x2032xf64>) -> tensor<130x2033x2032xf64>
  return %concat : tensor<130x2033x2032xf64>
}


// CHECK: func.func @test_pad_leftover(%arg0: tensor<128x2031x2032xf64>, %arg1: tensor<1x2032x2032xf64>, %arg2: tensor<1x2032x2032xf64>) -> tensor<130x2033x2032xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<128x2031x2032xf64>, tensor<f64>) -> tensor<128x2032x2032xf64>
// CHECK-NEXT:   %1 = stablehlo.concatenate %arg1, %0, %arg2, dim = 0 : (tensor<1x2032x2032xf64>, tensor<128x2032x2032xf64>, tensor<1x2032x2032xf64>) -> tensor<130x2032x2032xf64>
// CHECK-NEXT:   %2 = stablehlo.pad %1, %cst, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<130x2032x2032xf64>, tensor<f64>) -> tensor<130x2033x2032xf64>
// CHECK-NEXT:   return %2 : tensor<130x2033x2032xf64>
// CHECK-NEXT: }

func.func @test_pad_clean(%arg0 : tensor<128x2032x2032xf64>, %arg1 : tensor<1x2032x2032xf64>, %arg2: tensor<1x2032x2032xf64>) -> tensor<130x2033x2032xf64> {
  %cst_29 = stablehlo.constant dense<0.5> : tensor<f64>
  %p1 = stablehlo.pad %arg0, %cst_29, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<128x2032x2032xf64>, tensor<f64>) -> tensor<128x2033x2032xf64>
  %p2 = stablehlo.pad %arg1, %cst_29, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2032x2032xf64>, tensor<f64>) -> tensor<1x2033x2032xf64> 
  %p3 = stablehlo.pad %arg2, %cst_29, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2032x2032xf64>, tensor<f64>) -> tensor<1x2033x2032xf64> 

  %concat = stablehlo.concatenate %p2,%p1,%p3, dim = 0 : (tensor<1x2033x2032xf64>, tensor<128x2033x2032xf64>, tensor<1x2033x2032xf64>) -> tensor<130x2033x2032xf64>
  return %concat : tensor<130x2033x2032xf64>
}


// CHECK-NEXT: func.func @test_pad_clean(%arg0: tensor<128x2032x2032xf64>, %arg1: tensor<1x2032x2032xf64>, %arg2: tensor<1x2032x2032xf64>) -> tensor<130x2033x2032xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.concatenate %arg1, %arg0, %arg2, dim = 0 : (tensor<1x2032x2032xf64>, tensor<128x2032x2032xf64>, tensor<1x2032x2032xf64>) -> tensor<130x2032x2032xf64>
// CHECK-NEXT:   %1 = stablehlo.pad %0, %cst, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<130x2032x2032xf64>, tensor<f64>) -> tensor<130x2033x2032xf64>
// CHECK-NEXT:   return %1 : tensor<130x2033x2032xf64>
  // CHECK-NEXT: }

// Padding between the operands along the concatenation dimension must stay on
// the operands (https://github.com/EnzymeAD/Reactant.jl/issues/3479).
func.func @test_pad_concat_dim(%arg0 : tensor<558x1xf32>, %arg1 : tensor<558x1xf32>) -> tensor<1120x1xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %p1 = stablehlo.pad %arg0, %cst, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<558x1xf32>, tensor<f32>) -> tensor<560x1xf32>
  %p2 = stablehlo.pad %arg1, %cst, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<558x1xf32>, tensor<f32>) -> tensor<560x1xf32>
  %concat = stablehlo.concatenate %p1, %p2, dim = 0 : (tensor<560x1xf32>, tensor<560x1xf32>) -> tensor<1120x1xf32>
  return %concat : tensor<1120x1xf32>
}

// CHECK: func.func @test_pad_concat_dim(%arg0: tensor<558x1xf32>, %arg1: tensor<558x1xf32>) -> tensor<1120x1xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.pad %arg0, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<558x1xf32>, tensor<f32>) -> tensor<559x1xf32>
// CHECK-NEXT:   %1 = stablehlo.pad %arg1, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<558x1xf32>, tensor<f32>) -> tensor<559x1xf32>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<559x1xf32>, tensor<559x1xf32>) -> tensor<1118x1xf32>
// CHECK-NEXT:   %3 = stablehlo.pad %2, %cst, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<1118x1xf32>, tensor<f32>) -> tensor<1120x1xf32>
// CHECK-NEXT:   return %3 : tensor<1120x1xf32>
// CHECK-NEXT: }

// Only the first operand's low padding and the last operand's high padding
// along the concatenation dimension are hoisted; the other dimension uses the
// common minimum.
func.func @test_pad_concat_dim_three(%arg0 : tensor<4x8xf32>, %arg1 : tensor<5x8xf32>, %arg2 : tensor<6x8xf32>) -> tensor<24x10xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %p1 = stablehlo.pad %arg0, %cst, low = [2, 1], high = [1, 1], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<7x10xf32>
  %p2 = stablehlo.pad %arg1, %cst, low = [0, 1], high = [3, 1], interior = [0, 0] : (tensor<5x8xf32>, tensor<f32>) -> tensor<8x10xf32>
  %p3 = stablehlo.pad %arg2, %cst, low = [1, 1], high = [2, 1], interior = [0, 0] : (tensor<6x8xf32>, tensor<f32>) -> tensor<9x10xf32>
  %concat = stablehlo.concatenate %p1, %p2, %p3, dim = 0 : (tensor<7x10xf32>, tensor<8x10xf32>, tensor<9x10xf32>) -> tensor<24x10xf32>
  return %concat : tensor<24x10xf32>
}

// CHECK: func.func @test_pad_concat_dim_three(%arg0: tensor<4x8xf32>, %arg1: tensor<5x8xf32>, %arg2: tensor<6x8xf32>) -> tensor<24x10xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.pad %arg0, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<5x8xf32>
// CHECK-NEXT:   %1 = stablehlo.pad %arg1, %cst, low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<5x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:   %2 = stablehlo.pad %arg2, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<6x8xf32>, tensor<f32>) -> tensor<7x8xf32>
// CHECK-NEXT:   %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<5x8xf32>, tensor<8x8xf32>, tensor<7x8xf32>) -> tensor<20x8xf32>
// CHECK-NEXT:   %4 = stablehlo.pad %3, %cst, low = [2, 1], high = [2, 1], interior = [0, 0] : (tensor<20x8xf32>, tensor<f32>) -> tensor<24x10xf32>
// CHECK-NEXT:   return %4 : tensor<24x10xf32>
// CHECK-NEXT: }

// Nothing can be hoisted when all padding lies between the operands.
func.func @test_pad_concat_dim_inner_only(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<10xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %p1 = stablehlo.pad %arg0, %cst, low = [0], high = [1], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>
  %p2 = stablehlo.pad %arg1, %cst, low = [1], high = [0], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>
  %concat = stablehlo.concatenate %p1, %p2, dim = 0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  return %concat : tensor<10xf32>
}

// CHECK: func.func @test_pad_concat_dim_inner_only(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<10xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.pad %arg0, %cst, low = [0], high = [1], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK-NEXT:   %1 = stablehlo.pad %arg1, %cst, low = [1], high = [0], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
// CHECK-NEXT:   return %2 : tensor<10xf32>
// CHECK-NEXT: }
