// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck %s

func.func @while(%arg0: tensor<f64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<1.000000e+01> : tensor<f64>
  %0:2 = stablehlo.while(%iterArg0 = %arg0, %iterArg1 = %cst) : tensor<f64>, tensor<f64>
    cond {
      %1 = stablehlo.compare LT, %iterArg0, %iterArg1 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg0, %iterArg0 : tensor<f64>
      stablehlo.return %1, %iterArg1 : tensor<f64>, tensor<f64>
    }
  return %0#0 : tensor<f64>
}

// CHECK-LABEL: func.func @while
// CHECK: %[[ARG_TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<f32>, tensor<f32>>
// CHECK-DAG: %[[CST_HI:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<f32>
// CHECK-DAG: %[[CST_LO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[CST_TUPLE:.*]] = stablehlo.tuple %[[CST_HI]], %[[CST_LO]] : tuple<tensor<f32>, tensor<f32>>
// CHECK-DAG: %[[ARG_HI:.*]] = stablehlo.get_tuple_element %[[ARG_TUPLE]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK-DAG: %[[ARG_LO:.*]] = stablehlo.get_tuple_element %[[ARG_TUPLE]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK: stablehlo.while
// CHECK-NEXT: cond {
// CHECK: stablehlo.compare {{.*}}, %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: } do {
// CHECK:        stablehlo.return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
// CHECK-NEXT: }
