// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dusduspadpad_to_duspad" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

func.func @f(%iterArg_177 : tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>) {
  %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f64>

  %c_169 = stablehlo.constant dense<7> : tensor<i32> 
  %c_171 = stablehlo.constant dense<8> : tensor<i32>
  %c_172 = stablehlo.constant dense<0> : tensor<i32> 

  %503 = stablehlo.slice %iterArg_177 [8:12, 9:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6127x12272xf64>

  %504 = stablehlo.pad %503, %cst_161, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x6127x12272xf64>, tensor<f64>) -> tensor<4x6129x12272xf64>
  %505 = stablehlo.dynamic_update_slice %iterArg_177, %504, %c_171, %c_171, %c_172 : (tensor<20x6144x12272xf64>, tensor<4x6129x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

  %506 = stablehlo.slice %505 [8:12, 7:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
  %509 = stablehlo.slice %505 [8:12, 10:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>

  %2533 = stablehlo.slice %505 [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
  %2535 = stablehlo.slice %505 [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
  %2539 = stablehlo.slice %505 [8:12, 9:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>

  %512 = "enzymexla.extend"(%503) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x6127x12272xf64>) -> tensor<6x6127x12272xf64>

  %513 = stablehlo.pad %512, %cst_161, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<6x6127x12272xf64>, tensor<f64>) -> tensor<6x6128x12272xf64>
  %514 = stablehlo.dynamic_update_slice %505, %513, %c_169, %c_171, %c_172 : (tensor<20x6144x12272xf64>, tensor<6x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

  func.return %514, %506, %509, %2533, %2535, %2539 : tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>
}

// CHECK:  func.func @f(%arg0: tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-DAG:     %[[SLICE:.+]] = stablehlo.slice %arg0 [8:12, 9:6136, 0:12272]
// CHECK-DAG:     %[[EXT:.+]] = "enzymexla.extend"(%[[SLICE]]) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}>
// CHECK-DAG:     %[[PAD:.+]] = stablehlo.pad %[[EXT]], %cst
// CHECK-NEXT:    %[[DUS:.+]] = stablehlo.dynamic_update_slice %arg0, %[[PAD]], %c, %c_0, %c_1
// CHECK-NEXT:    %[[S1:.+]] = stablehlo.slice %[[DUS]] [8:12, 7:6135, 0:12272]
// CHECK-NEXT:    %[[S2:.+]] = stablehlo.slice %[[DUS]] [8:12, 10:6138, 0:12272]
// CHECK-NEXT:    %[[S3:.+]] = stablehlo.slice %[[DUS]] [8:12, 6:6135, 0:12272]
// CHECK-NEXT:    %[[S4:.+]] = stablehlo.slice %[[DUS]] [8:12, 7:6136, 0:12272]
// CHECK-NEXT:    %[[S5:.+]] = stablehlo.slice %[[DUS]] [8:12, 9:6138, 0:12272]
// CHECK-NEXT:    return %[[DUS]], %[[S1]], %[[S2]], %[[S3]], %[[S4]], %[[S5]]
// CHECK-NEXT:  }
