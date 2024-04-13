// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @t1(%1383: tensor<1x30x1x10xbf16>, %1387: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1384 = stablehlo.pad %1383, %cst_217, low = [0, 0, 0, 10], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x20xbf16>
  %1388 = stablehlo.pad %1387, %cst_217, low = [0, 0, 0, 0], high = [0, 0, 0, 10], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x20xbf16>
  %1389 = stablehlo.add %1384, %1388 : tensor<1x30x1x20xbf16> 
  return %1389 : tensor<1x30x1x20xbf16>
}


func.func @t2(%1383: tensor<1x30x1x10xbf16>, %1387: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1384 = stablehlo.pad %1383, %cst_217, low = [0, 0, 0, 0], high = [0, 0, 0, 10], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x20xbf16>
  %1388 = stablehlo.pad %1387, %cst_217, low = [0, 0, 0, 10], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x20xbf16>
  %1389 = stablehlo.add %1384, %1388 : tensor<1x30x1x20xbf16> 
  return %1389 : tensor<1x30x1x20xbf16>
}

// CHECK:  func.func @t1(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    return %0 : tensor<1x30x1x20xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @t2(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    return %0 : tensor<1x30x1x20xbf16>
// CHECK-NEXT:  }
