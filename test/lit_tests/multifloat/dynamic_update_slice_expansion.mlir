// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=CHECK-LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=CHECK-TUPLE %s

func.func @dus_f64(%operand: tensor<4x4xf64>, %update: tensor<2x2xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<4x4xf64> {
  // CHECK-LABEL: @dus_f64
  // CHECK-NOT: stablehlo.slice
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RES:.*]] = stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[C0]], %{{.*}}, %{{.*}} : (tensor<2x4x4xf32>, tensor<2x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4xf32>
  // CHECK-NOT: stablehlo.concatenate
  // CHECK: %[[CONV:.*]] = stablehlo.convert %[[RES]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
  // CHECK: %[[REDUCE:.*]] = stablehlo.reduce(%[[CONV]] init: %{{.*}}) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
  // CHECK: return %[[REDUCE]] : tensor<4x4xf64>

  // CHECK-LAST-LABEL: @dus_f64
  // CHECK-LAST-NOT: stablehlo.slice
  // CHECK-LAST: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-LAST: %[[RES:.*]] = stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C0]] : (tensor<4x4x2xf32>, tensor<2x2x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x2xf32>
  // CHECK-LAST-NOT: stablehlo.concatenate
  // CHECK-LAST: %[[CONV:.*]] = stablehlo.convert %[[RES]] : (tensor<4x4x2xf32>) -> tensor<4x4x2xf64>
  // CHECK-LAST: %[[REDUCE:.*]] = stablehlo.reduce(%[[CONV]] init: %{{.*}}) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
  // CHECK-LAST: return %[[REDUCE]] : tensor<4x4xf64>

  // CHECK-TUPLE-LABEL: @dus_f64
  // CHECK-TUPLE: %[[OP_HI:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
  // CHECK-TUPLE: %[[OP_LO:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
  // CHECK-TUPLE: %[[UPD_HI:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
  // CHECK-TUPLE: %[[UPD_LO:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<2x2xf32>, tensor<2x2xf32>>) -> tensor<2x2xf32>
  // CHECK-TUPLE-NOT: stablehlo.reshape
  // CHECK-TUPLE: %[[RES_HI:.*]] = stablehlo.dynamic_update_slice %[[OP_HI]], %[[UPD_HI]], %{{.*}}, %{{.*}} : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // CHECK-TUPLE: %[[RES_LO:.*]] = stablehlo.dynamic_update_slice %[[OP_LO]], %[[UPD_LO]], %{{.*}}, %{{.*}} : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // CHECK-TUPLE-NOT: stablehlo.reshape
  // CHECK-TUPLE: %[[CONV_HI:.*]] = stablehlo.convert %[[RES_HI]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
  // CHECK-TUPLE: %[[CONV_LO:.*]] = stablehlo.convert %[[RES_LO]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
  // CHECK-TUPLE: %[[RET:.*]] = stablehlo.add %[[CONV_HI]], %[[CONV_LO]] : tensor<4x4xf64>
  // CHECK-TUPLE: return %[[RET]] : tensor<4x4xf64>

  %0 = stablehlo.dynamic_update_slice %operand, %update, %i, %j : (tensor<4x4xf64>, tensor<2x2xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
