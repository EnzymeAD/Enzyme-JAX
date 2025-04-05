// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=const_pad_concat_to_concat},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

func.func @test_pad_low(%arg0 : tensor<1x2031x4080xf64>, %arg1 : tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %concat = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2031x4080xf64>, tensor<1x1x4080xf64>) -> tensor<1x2032x4080xf64>
    %pad = stablehlo.pad %concat, %cst, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2032x4080xf64>, tensor<f64>) -> tensor<1x2033x4080xf64>
    return %pad : tensor<1x2033x4080xf64>
}

// CHECK: func.func @test_pad_low(%arg0: tensor<1x2031x4080xf64>, %arg1: tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x1x4080xf64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %cst, %arg0, %arg1, dim = 1 : (tensor<1x1x4080xf64>, tensor<1x2031x4080xf64>, tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64>
// CHECK-NEXT:    return %0 : tensor<1x2033x4080xf64>
// CHECK-NEXT:    }


func.func @test_pad_high(%arg0 : tensor<1x2031x4080xf64>, %arg1 : tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %concat = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2031x4080xf64>, tensor<1x1x4080xf64>) -> tensor<1x2032x4080xf64>
    %pad = stablehlo.pad %concat, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<1x2032x4080xf64>, tensor<f64>) -> tensor<1x2033x4080xf64>
    return %pad : tensor<1x2033x4080xf64>
}

// CHECK: func.func @test_pad_high(%arg0: tensor<1x2031x4080xf64>, %arg1: tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x1x4080xf64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, %cst, dim = 1 : (tensor<1x2031x4080xf64>, tensor<1x1x4080xf64>, tensor<1x1x4080xf64>) -> tensor<1x2033x4080xf64>
// CHECK-NEXT:    return %0 : tensor<1x2033x4080xf64>
// CHECK-NEXT:  }

func.func @test_pad_high2(%arg0 : tensor<1x2031x4080xf64>, %arg1 : tensor<1x2031x4080xf64>) -> tensor<3x2031x4080xf64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %concat = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1x2031x4080xf64>, tensor<1x2031x4080xf64>) -> tensor<2x2031x4080xf64>
    %pad = stablehlo.pad %concat, %cst, low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<2x2031x4080xf64>, tensor<f64>) -> tensor<3x2031x4080xf64>
    return %pad : tensor<3x2031x4080xf64>
}

// CHECK:    func.func @test_pad_high2(%arg0: tensor<1x2031x4080xf64>, %arg1: tensor<1x2031x4080xf64>) -> tensor<3x2031x4080xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x2031x4080xf64>
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, %cst, dim = 0 : (tensor<1x2031x4080xf64>, tensor<1x2031x4080xf64>, tensor<1x2031x4080xf64>) -> tensor<3x2031x4080xf64>
// CHECK-NEXT:    return %0 : tensor<3x2031x4080xf64>
// CHECK-NEXT:  }
