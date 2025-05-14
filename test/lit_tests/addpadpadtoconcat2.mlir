// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=add_pad_pad_to_concat;pad_simplify(1024)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

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

func.func @t3(%1383: tensor<1x30x1x10xbf16>, %1387: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1384 = stablehlo.pad %1383, %cst_217, low = [0, 0, 0, 90], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1388 = stablehlo.pad %1387, %cst_217, low = [0, 0, 0, 80], high = [0, 0, 0, 10], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1389 = stablehlo.add %1384, %1388 : tensor<1x30x1x100xbf16> 
  return %1389 : tensor<1x30x1x100xbf16>
}

func.func @t4(%1383: tensor<1x30x1x10xbf16>, %1387: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1384 = stablehlo.pad %1383, %cst_217, low = [0, 0, 0, 90], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1388 = stablehlo.pad %1387, %cst_217, low = [0, 0, 0, 80], high = [0, 0, 0, 10], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1389 = stablehlo.add %1388, %1384 : tensor<1x30x1x100xbf16> 
  return %1389 : tensor<1x30x1x100xbf16>
}

func.func @t5(%1383: tensor<1x30x1x10xbf16>, %1387: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1384 = stablehlo.pad %1383, %cst_217, low = [0, 0, 0, 30], high = [0, 0, 0, 60], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1388 = stablehlo.pad %1387, %cst_217, low = [0, 0, 0, 20], high = [0, 0, 0, 70], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
  %1389 = stablehlo.add %1388, %1384 : tensor<1x30x1x100xbf16> 
  return %1389 : tensor<1x30x1x100xbf16>
}

func.func @t6(%1301: tensor<1x1x4096x1x256xbf16>, %1303: tensor<1x1x2048x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1302 = "stablehlo.pad"(%1301, %cst_217) {edge_padding_high = array<i64: 0, 0, 4096, 0, 0>, edge_padding_low = array<i64: 0, 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0, 0>} : (tensor<1x1x4096x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
  %1304 = "stablehlo.pad"(%1303, %cst_217) {edge_padding_high = array<i64: 0, 0, 6144, 0, 0>, edge_padding_low = array<i64: 0, 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0, 0>} : (tensor<1x1x2048x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
  %1101 = stablehlo.add %1302, %1304 : tensor<1x1x8192x1x256xbf16>
  return %1101 : tensor<1x1x8192x1x256xbf16>
}

func.func @t7(%1301: tensor<1x1x4096x1x256xbf16>, %1303: tensor<1x1x2048x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> {
  %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<bf16> 
  %1302 = "stablehlo.pad"(%1301, %cst_217) {edge_padding_high = array<i64: 0, 0, 4096, 0, 0>, edge_padding_low = array<i64: 0, 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0, 0>} : (tensor<1x1x4096x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
  %1304 = "stablehlo.pad"(%1303, %cst_217) {edge_padding_high = array<i64: 0, 0, 6144, 0, 0>, edge_padding_low = array<i64: 0, 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0, 0>} : (tensor<1x1x2048x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
  %1101 = stablehlo.add %1304, %1302 : tensor<1x1x8192x1x256xbf16>
  return %1101 : tensor<1x1x8192x1x256xbf16>
}

// CHECK:  func.func @t1(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    return %0 : tensor<1x30x1x20xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @t2(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    return %0 : tensor<1x30x1x20xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @t3(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x20xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %[[i2]] : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @t4(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x20xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %[[i2]] : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @t5(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %[[i0]], low = [0, 0, 0, 20], high = [0, 0, 0, 60], interior = [0, 0, 0, 0] : (tensor<1x30x1x20xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %[[i2]] : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

//CHECK:  func.func @t6(%arg0: tensor<1x1x4096x1x256xbf16>, %arg1: tensor<1x1x2048x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> {
//CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
//CHECK-NEXT:    %[[i1:.+]] = stablehlo.pad %arg1, %[[i0]], low = [0, 0, 0, 0, 0], high = [0, 0, 2048, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x2048x1x256xbf16>, tensor<bf16>) -> tensor<1x1x4096x1x256xbf16>
//CHECK-NEXT:    %[[i2:.+]] = stablehlo.add %[[i1]], %arg0 : tensor<1x1x4096x1x256xbf16>
//CHECK-NEXT:    %[[i3:.+]] = stablehlo.pad %[[i2]], %[[i0]], low = [0, 0, 0, 0, 0], high = [0, 0, 4096, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x4096x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
//CHECK-NEXT:    return %[[i3]] : tensor<1x1x8192x1x256xbf16>
//CHECK-NEXT:  }

// CHECK:  func.func @t7(%arg0: tensor<1x1x4096x1x256xbf16>, %arg1: tensor<1x1x2048x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.pad %arg1, %[[i0]], low = [0, 0, 0, 0, 0], high = [0, 0, 2048, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x2048x1x256xbf16>, tensor<bf16>) -> tensor<1x1x4096x1x256xbf16>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.add %[[i1]], %arg0 : tensor<1x1x4096x1x256xbf16>
// CHECK-NEXT:    %[[i3:.+]] = stablehlo.pad %[[i2]], %[[i0]], low = [0, 0, 0, 0, 0], high = [0, 0, 4096, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x4096x1x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x1x256xbf16>
// CHECK-NEXT:    return %[[i3]] : tensor<1x1x8192x1x256xbf16>
// CHECK-NEXT:  }
