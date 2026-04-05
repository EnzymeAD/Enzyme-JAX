// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s

func.func @add(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

// FIRST-LABEL: func.func @add
// FIRST:     %[[V_0:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<f64> to tensor<f32>
// FIRST:     %[[V_1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<f64> to tensor<f32>
// FIRST:     %[[V_2:.*]] = stablehlo.add %[[V_1]], %[[V_0]] : tensor<f32>
// FIRST:     %[[V_3:.*]] = builtin.unrealized_conversion_cast %[[V_2]] : tensor<f32> to tensor<f64>
// FIRST:     return %[[V_3]] : tensor<f64>

// LAST-LABEL: func.func @add
// LAST:     %[[V_0:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<f64> to tensor<f32>
// LAST:     %[[V_1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<f64> to tensor<f32>
// LAST:     %[[V_2:.*]] = stablehlo.add %[[V_1]], %[[V_0]] : tensor<f32>
// LAST:     %[[V_3:.*]] = builtin.unrealized_conversion_cast %[[V_2]] : tensor<f32> to tensor<f64>
// LAST:     return %[[V_3]] : tensor<f64>

// TUPLE-LABEL: func.func @add
// TUPLE:     %[[V_0:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<f64> to tensor<f32>
// TUPLE:     %[[V_1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<f64> to tensor<f32>
// TUPLE:     %[[V_2:.*]] = stablehlo.add %[[V_1]], %[[V_0]] : tensor<f32>
// TUPLE:     %[[V_3:.*]] = builtin.unrealized_conversion_cast %[[V_2]] : tensor<f32> to tensor<f64>
// TUPLE:     return %[[V_3]] : tensor<f64>
