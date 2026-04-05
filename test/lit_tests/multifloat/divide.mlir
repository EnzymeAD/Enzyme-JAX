// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s

func.func @divide(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-LABEL: @divide
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<2xf64>
// CHECK: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[X_HI:.*]] = stablehlo.slice %[[CAT]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[X_LO:.*]] = stablehlo.slice %[[CAT]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[Y_HI:.*]] = stablehlo.slice %[[CAT2]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[Y_LO:.*]] = stablehlo.slice %[[CAT2]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1x2xf32>
// CHECK: %[[U:.*]] = stablehlo.divide %[[ONE]], %[[Y_HI]] : tensor<1x2xf32>
// CHECK: %[[Q_HI:.*]] = stablehlo.multiply %[[X_HI]], %[[U]] : tensor<1x2xf32>
// CHECK: %[[CST3:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[X_HI_BIG:.*]] = stablehlo.multiply %[[X_HI]], %[[CST3]] : tensor<1x2xf32>
// CHECK: %[[T:.*]] = stablehlo.subtract %[[X_HI_BIG]], %[[X_HI]] : tensor<1x2xf32>
// CHECK: %[[X_HI_HI:.*]] = stablehlo.subtract %[[X_HI_BIG]], %[[T]] : tensor<1x2xf32>
// CHECK: %[[X_HI_LO:.*]] = stablehlo.subtract %[[X_HI]], %[[X_HI_HI]] : tensor<1x2xf32>
// CHECK: %[[CST4:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[U_BIG:.*]] = stablehlo.multiply %[[U]], %[[CST4]] : tensor<1x2xf32>
// CHECK: %[[T_U:.*]] = stablehlo.subtract %[[U_BIG]], %[[U]] : tensor<1x2xf32>
// CHECK: %[[U_HI:.*]] = stablehlo.subtract %[[U_BIG]], %[[T_U]] : tensor<1x2xf32>
// CHECK: %[[U_LO:.*]] = stablehlo.subtract %[[U]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P0:.*]] = stablehlo.multiply %[[X_HI_HI]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P1:.*]] = stablehlo.multiply %[[X_HI_HI]], %[[U_LO]] : tensor<1x2xf32>
// CHECK: %[[P2:.*]] = stablehlo.multiply %[[X_HI_LO]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P3:.*]] = stablehlo.multiply %[[X_HI_LO]], %[[U_LO]] : tensor<1x2xf32>
  // CHECK: return
  
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
