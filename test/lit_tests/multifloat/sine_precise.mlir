// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s

func.func @sine(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sine %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @sine
// TUPLE:    %[[CVT0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:    %[[CVT1:.*]] = stablehlo.convert %[[CVT0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:    %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[CVT1]] : tensor<2xf64>
// TUPLE:    %[[CVT3:.*]] = stablehlo.convert %[[SUB1]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:    %[[TUPLE4:.*]] = stablehlo.tuple %[[CVT0]], %[[CVT3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:    %[[HI:.*]] = stablehlo.get_tuple_element %[[TUPLE4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:    %[[LO:.*]] = stablehlo.get_tuple_element %[[TUPLE4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:    %[[INV_PI_HI:.*]] = stablehlo.constant dense<0.318309873> : tensor<2xf32>
// TUPLE:    %[[INV_PI_LO:.*]] = stablehlo.constant dense<1.28412765E-8> : tensor<2xf32>
// TUPLE:    %[[MUL7:.*]] = stablehlo.multiply %[[HI]], %[[INV_PI_HI]] : tensor<2xf32>
// TUPLE:    %{{.*}} = stablehlo.constant dense<4.097000e+03> : tensor<2xf32>
// TUPLE:    %[[MUL8:.*]] = stablehlo.multiply %[[HI]], %{{.*}} : tensor<2xf32>
// TUPLE:    %[[SUB9:.*]] = stablehlo.subtract %[[MUL8]], %[[HI]] : tensor<2xf32>
// TUPLE:    %[[SUB10:.*]] = stablehlo.subtract %[[MUL8]], %[[SUB9]] : tensor<2xf32>
// TUPLE:    %[[SUB11:.*]] = stablehlo.subtract %[[HI]], %[[SUB10]] : tensor<2xf32>
// TUPLE:    %{{.*}} = stablehlo.multiply %[[INV_PI_HI]], %{{.*}} : tensor<2xf32>

// FIRST-LABEL: func.func @sine
// FIRST:    %[[CVT0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:    %[[CVT1:.*]] = stablehlo.convert %[[CVT0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:    %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[CVT1]] : tensor<2xf64>
// FIRST:    %[[CVT3:.*]] = stablehlo.convert %[[SUB1]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:    %[[RESHAPE0:.*]] = stablehlo.reshape %[[CVT0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:    %[[RESHAPE1:.*]] = stablehlo.reshape %[[CVT3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:    %[[CONCAT:.*]] = stablehlo.concatenate %[[RESHAPE0]], %[[RESHAPE1]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:    %[[HI:.*]] = stablehlo.slice %[[CONCAT]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:    %[[LO:.*]] = stablehlo.slice %[[CONCAT]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:    %[[INV_PI_HI:.*]] = stablehlo.constant dense<0.318309873> : tensor<1x2xf32>
// FIRST:    %[[INV_PI_LO:.*]] = stablehlo.constant dense<1.28412765E-8> : tensor<1x2xf32>
// FIRST:    %[[X_PI_HI:.*]] = stablehlo.multiply %[[HI]], %[[INV_PI_HI]] : tensor<1x2xf32>
// FIRST:    %[[CST3:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// FIRST:    %[[HI_BIG:.*]] = stablehlo.multiply %[[HI]], %[[CST3]] : tensor<1x2xf32>
// FIRST:    %[[T:.*]] = stablehlo.subtract %[[HI_BIG]], %[[HI]] : tensor<1x2xf32>
// FIRST:    %[[HI_HI:.*]] = stablehlo.subtract %[[HI_BIG]], %[[T]] : tensor<1x2xf32>
// FIRST:    %[[HI_LO:.*]] = stablehlo.subtract %[[HI]], %[[HI_HI]] : tensor<1x2xf32>
// FIRST:    %[[CST4:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// FIRST:    %[[INV_PI_BIG:.*]] = stablehlo.multiply %[[INV_PI_HI]], %[[CST4]] : tensor<1x2xf32>
// FIRST:    %[[T_U:.*]] = stablehlo.subtract %[[INV_PI_BIG]], %[[INV_PI_HI]] : tensor<1x2xf32>
// FIRST:    %[[INV_PI_HI_HI:.*]] = stablehlo.subtract %[[INV_PI_BIG]], %[[T_U]] : tensor<1x2xf32>
// FIRST:    %[[INV_PI_HI_LO:.*]] = stablehlo.subtract %[[INV_PI_HI]], %[[INV_PI_HI_HI]] : tensor<1x2xf32>
// FIRST:    %[[P0:.*]] = stablehlo.multiply %[[HI_HI]], %[[INV_PI_HI_HI]] : tensor<1x2xf32>
// FIRST:    %[[P1:.*]] = stablehlo.multiply %[[HI_HI]], %[[INV_PI_HI_LO]] : tensor<1x2xf32>
// FIRST:    %[[P2:.*]] = stablehlo.multiply %[[HI_LO]], %[[INV_PI_HI_HI]] : tensor<1x2xf32>
// FIRST:    %[[P3:.*]] = stablehlo.multiply %[[HI_LO]], %[[INV_PI_HI_LO]] : tensor<1x2xf32>
// FIRST:    %[[E0:.*]] = stablehlo.subtract %[[P0]], %[[X_PI_HI]] : tensor<1x2xf32>
// FIRST:    %[[E1:.*]] = stablehlo.add %[[E0]], %[[P1]] : tensor<1x2xf32>
// FIRST:    %[[E2:.*]] = stablehlo.add %[[E1]], %[[P2]] : tensor<1x2xf32>
// FIRST:    %[[E3:.*]] = stablehlo.add %[[E2]], %[[P3]] : tensor<1x2xf32>
