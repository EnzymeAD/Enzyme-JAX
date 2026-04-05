// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

func.func @sqrt(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sqrt %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @sqrt(%arg0: tensor<2xf64>) -> tensor<2xf64> {
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[X_HI:.*]] = stablehlo.slice %[[CAT]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[X_LO:.*]] = stablehlo.slice %[[CAT]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1x2xf32>
// CHECK: %[[IS_ZERO:.*]] = stablehlo.compare EQ, %[[X_HI]], %[[ZERO]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// CHECK: %[[SAFE_X:.*]] = stablehlo.select %[[IS_ZERO]], %[[ONE]], %[[X_HI]] : tensor<1x2xi1>, tensor<1x2xf32>
// CHECK: %[[U0:.*]] = stablehlo.rsqrt %[[SAFE_X]] : tensor<1x2xf32>
// CHECK: %[[ROOT_HI:.*]] = stablehlo.multiply %[[X_HI]], %[[U0]] : tensor<1x2xf32>
// CHECK: %[[CST3:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[X_HI_BIG:.*]] = stablehlo.multiply %[[X_HI]], %[[CST3]] : tensor<1x2xf32>
// CHECK: %[[T:.*]] = stablehlo.subtract %[[X_HI_BIG]], %[[X_HI]] : tensor<1x2xf32>
// CHECK: %[[X_HI_HI:.*]] = stablehlo.subtract %[[X_HI_BIG]], %[[T]] : tensor<1x2xf32>
// CHECK: %[[X_HI_LO:.*]] = stablehlo.subtract %[[X_HI]], %[[X_HI_HI]] : tensor<1x2xf32>
// CHECK: %[[CST4:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[U_BIG:.*]] = stablehlo.multiply %[[U0]], %[[CST4]] : tensor<1x2xf32>
// CHECK: %[[T_U:.*]] = stablehlo.subtract %[[U_BIG]], %[[U0]] : tensor<1x2xf32>
// CHECK: %[[U_HI:.*]] = stablehlo.subtract %[[U_BIG]], %[[T_U]] : tensor<1x2xf32>
// CHECK: %[[U_LO:.*]] = stablehlo.subtract %[[U0]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P0:.*]] = stablehlo.multiply %[[X_HI_HI]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P1:.*]] = stablehlo.multiply %[[X_HI_HI]], %[[U_LO]] : tensor<1x2xf32>
// CHECK: %[[P2:.*]] = stablehlo.multiply %[[X_HI_LO]], %[[U_HI]] : tensor<1x2xf32>
// CHECK: %[[P3:.*]] = stablehlo.multiply %[[X_HI_LO]], %[[U_LO]] : tensor<1x2xf32>
// CHECK: %[[E0:.*]] = stablehlo.subtract %[[P0]], %[[ROOT_HI]] : tensor<1x2xf32>
// CHECK: %[[E1:.*]] = stablehlo.add %[[E0]], %[[P1]] : tensor<1x2xf32>
// CHECK: %[[E2:.*]] = stablehlo.add %[[E1]], %[[P2]] : tensor<1x2xf32>
// CHECK: %[[E3:.*]] = stablehlo.add %[[E2]], %[[P3]] : tensor<1x2xf32>
// CHECK: %[[P4:.*]] = stablehlo.multiply %[[X_HI]], %[[ZERO]] : tensor<1x2xf32>
// CHECK: %[[P5:.*]] = stablehlo.multiply %[[X_LO]], %[[U0]] : tensor<1x2xf32>
// CHECK: %[[ADD1:.*]] = stablehlo.add %[[P4]], %[[P5]] : tensor<1x2xf32>
// CHECK: %[[ADD2:.*]] = stablehlo.add %[[E3]], %[[ADD1]] : tensor<1x2xf32>
// CHECK: %[[ROOT_FINAL:.*]] = stablehlo.add %[[ROOT_HI]], %[[ADD2]] : tensor<1x2xf32>
// CHECK: %[[RES1:.*]] = stablehlo.subtract %[[ROOT_FINAL]], %[[ROOT_HI]] : tensor<1x2xf32>
// CHECK: %[[CORR1:.*]] = stablehlo.subtract %[[ADD2]], %[[RES1]] : tensor<1x2xf32>
// CHECK: %[[ROOT_SQ:.*]] = stablehlo.multiply %[[ROOT_FINAL]], %[[ROOT_FINAL]] : tensor<1x2xf32>
// CHECK: %[[CST5:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[ROOT_BIG:.*]] = stablehlo.multiply %[[ROOT_FINAL]], %[[CST5]] : tensor<1x2xf32>
// CHECK: %[[T2:.*]] = stablehlo.subtract %[[ROOT_BIG]], %[[ROOT_FINAL]] : tensor<1x2xf32>
// CHECK: %[[ROOT_F_HI:.*]] = stablehlo.subtract %[[ROOT_BIG]], %[[T2]] : tensor<1x2xf32>
// CHECK: %[[ROOT_F_LO:.*]] = stablehlo.subtract %[[ROOT_FINAL]], %[[ROOT_F_HI]] : tensor<1x2xf32>
// CHECK: %[[CST6:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
// CHECK: %[[ROOT_BIG2:.*]] = stablehlo.multiply %[[ROOT_FINAL]], %[[CST6]] : tensor<1x2xf32>
// CHECK: %[[T3:.*]] = stablehlo.subtract %[[ROOT_BIG2]], %[[ROOT_FINAL]] : tensor<1x2xf32>
// CHECK: %[[ROOT_F_HI2:.*]] = stablehlo.subtract %[[ROOT_BIG2]], %[[T3]] : tensor<1x2xf32>
// CHECK: %[[ROOT_F_LO2:.*]] = stablehlo.subtract %[[ROOT_FINAL]], %[[ROOT_F_HI2]] : tensor<1x2xf32>
// CHECK: %[[P6:.*]] = stablehlo.multiply %[[ROOT_F_HI]], %[[ROOT_F_HI2]] : tensor<1x2xf32>
// CHECK: %[[P7:.*]] = stablehlo.multiply %[[ROOT_F_HI]], %[[ROOT_F_LO2]] : tensor<1x2xf32>
// CHECK: %[[P8:.*]] = stablehlo.multiply %[[ROOT_F_LO]], %[[ROOT_F_HI2]] : tensor<1x2xf32>
// CHECK: %[[P9:.*]] = stablehlo.multiply %[[ROOT_F_LO]], %[[ROOT_F_LO2]] : tensor<1x2xf32>
// CHECK: %[[E4:.*]] = stablehlo.subtract %[[P6]], %[[ROOT_SQ]] : tensor<1x2xf32>
// CHECK: %[[E5:.*]] = stablehlo.add %[[E4]], %[[P7]] : tensor<1x2xf32>
// CHECK: %[[E6:.*]] = stablehlo.add %[[E5]], %[[P8]] : tensor<1x2xf32>
// CHECK: %[[E7:.*]] = stablehlo.add %[[E6]], %[[P9]] : tensor<1x2xf32>
// CHECK: return %{{.*}} : tensor<2xf64>

func.func @sqrt_zero() -> tensor<2xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
  %0 = stablehlo.sqrt %cst : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// CHECK: func @sqrt_zero
// CHECK: %[[ZERO2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// CHECK: %[[IS_ZERO2:.*]] = stablehlo.compare EQ, %{{.*}}, %[[ZERO2]]
// CHECK: %{{.*}} = stablehlo.select %[[IS_ZERO2]], %{{.*}}, %{{.*}}
// CHECK: %{{.*}} = stablehlo.select %[[IS_ZERO2]], %{{.*}}, %{{.*}}
