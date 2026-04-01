// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s

func.func @compare_f64_eq(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_eq
// FIRST: %[[CMP:.*]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// FIRST: %[[INIT:.*]] = stablehlo.constant dense<true> : tensor<i1>
// FIRST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.and across dimensions = [0] : (tensor<2x4x4xi1>, tensor<i1>) -> tensor<4x4xi1>
// FIRST: return %[[OUT]] : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_eq
// LAST: %[[CMP:.*]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4x4x2xf32>, tensor<4x4x2xf32>) -> tensor<4x4x2xi1>
// LAST: %[[INIT:.*]] = stablehlo.constant dense<true> : tensor<i1>
// LAST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.and across dimensions = [2] : (tensor<4x4x2xi1>, tensor<i1>) -> tensor<4x4xi1>
// LAST: return %[[OUT]] : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_eq
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_EQ:.*]] = stablehlo.compare EQ, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH_EQ]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW_EQ]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.and %[[HIGH_RESHAPED]], %[[LOW_RESHAPED]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>


func.func @compare_f64_ne(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_ne
// FIRST: %[[CMP:.*]] = stablehlo.compare NE, %{{.*}}, %{{.*}} : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// FIRST: %[[INIT:.*]] = stablehlo.constant dense<false> : tensor<i1>
// FIRST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.or across dimensions = [0] : (tensor<2x4x4xi1>, tensor<i1>) -> tensor<4x4xi1>
// FIRST: return %[[OUT]] : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_ne
// LAST: %[[CMP:.*]] = stablehlo.compare NE, %{{.*}}, %{{.*}} : (tensor<4x4x2xf32>, tensor<4x4x2xf32>) -> tensor<4x4x2xi1>
// LAST: %[[INIT:.*]] = stablehlo.constant dense<false> : tensor<i1>
// LAST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.or across dimensions = [2] : (tensor<4x4x2xi1>, tensor<i1>) -> tensor<4x4xi1>
// LAST: return %[[OUT]] : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_ne
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_NE:.*]] = stablehlo.compare NE, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_NE:.*]] = stablehlo.compare NE, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH_NE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW_NE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.or %[[HIGH_RESHAPED]], %[[LOW_RESHAPED]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>


func.func @compare_f64_ge(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_ge
// FIRST: %[[HIGH:.*]] = stablehlo.slice %{{.*}} [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[LOW:.*]] = stablehlo.slice %{{.*}} [1:2, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[HIGH_GT:.*]] = stablehlo.compare GT, %[[HIGH]], %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[LOW_GE:.*]] = stablehlo.compare GE, %[[LOW]], %{{.*}} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[OUT:.*]] = stablehlo.select %[[HIGH_EQ]], %[[LOW_GE]], %[[HIGH_GT]] : tensor<1x4x4xi1>
// FIRST: return %{{.*}} : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_ge
// LAST: %[[HIGH:.*]] = stablehlo.slice %{{.*}} [0:4, 0:4, 0:1] : (tensor<4x4x2xf32>) -> tensor<4x4x1xf32>
// LAST: %[[LOW:.*]] = stablehlo.slice %{{.*}} [0:4, 0:4, 1:2] : (tensor<4x4x2xf32>) -> tensor<4x4x1xf32>
// LAST: %[[HIGH_GT:.*]] = stablehlo.compare GT, %[[HIGH]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[LOW_GE:.*]] = stablehlo.compare GE, %[[LOW]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[OUT:.*]] = stablehlo.select %[[HIGH_EQ]], %[[LOW_GE]], %[[HIGH_GT]] : tensor<4x4x1xi1>
// LAST: return %{{.*}} : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_ge
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_GT:.*]] = stablehlo.compare GT, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_GE:.*]] = stablehlo.compare GE, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.select %[[HIGH_EQ]], %[[LOW_GE]], %[[HIGH_GT]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>
