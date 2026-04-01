// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | FileCheck %s --check-prefix=TUPLE

func.func @pad_zero(%arg0: tensor<2xf64>) -> tensor<4xf64> {
  // CHECK-LABEL: @pad_zero
  // CHECK: %[[C_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %{{.*}} = stablehlo.pad %{{.*}}, %[[C_ZERO]], low = [0, 1], high = [0, 1], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x4xf32>
  
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.pad %arg0, %cst, low = [1], high = [1], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

func.func @pad_tuple(%arg0: tensor<2xf64>) -> tensor<4xf64> {
  // TUPLE-LABEL: @pad_tuple
  // TUPLE: %[[HI_PAD:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // TUPLE: %[[LO_PAD:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // TUPLE: %[[HI_OP:.*]] = stablehlo.get_tuple_element %{{.*}}[0]
  // TUPLE: %[[LO_OP:.*]] = stablehlo.get_tuple_element %{{.*}}[1]
  // TUPLE: %[[HI_RES:.*]] = stablehlo.pad %[[HI_OP]], %[[HI_PAD]], low = [1], high = [1], interior = [0]
  // TUPLE: %[[LO_RES:.*]] = stablehlo.pad %[[LO_OP]], %[[LO_PAD]], low = [1], high = [1], interior = [0]
  // TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[HI_RES]], %[[LO_RES]]
  // TUPLE: return %{{.*}} : tensor<4xf64>
  
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.pad %arg0, %cst, low = [1], high = [1], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
