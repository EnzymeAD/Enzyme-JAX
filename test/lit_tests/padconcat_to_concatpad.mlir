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
