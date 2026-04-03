// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple convert-signatures=false" %s | FileCheck %s

module {
  func.func @main(%cond: tensor<i1>, %counter: tensor<i64>, %val: tensor<f64>) -> (tensor<i64>, tensor<f64>) {
    %0:2 = stablehlo.while(%iter_counter = %counter, %iter_val = %val) : tensor<i64>, tensor<f64>
      cond {
        stablehlo.return %cond : tensor<i1>
      } do {
        stablehlo.return %iter_counter, %iter_val : tensor<i64>, tensor<f64>
      }
    return %0#0, %0#1 : tensor<i64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[C1:.*]] = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[C1]] : (tensor<f32>) -> tensor<f64>
// CHECK: %{{.*}} = stablehlo.subtract %arg2, %[[C2]] : tensor<f64>
// CHECK: %[[TUPLE:.*]] = stablehlo.tuple %[[C1]], %{{.*}} : tuple<tensor<f32>, tensor<f32>>
// CHECK-DAG: %[[HI:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK-DAG: %[[LO:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// CHECK: stablehlo.while(%iterArg = %arg1, %iterArg_0 = %[[HI]], %iterArg_1 = %[[LO]]) : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT: cond {
// CHECK:        stablehlo.return %arg0 : tensor<i1>
// CHECK: } do {
// CHECK:        stablehlo.return %iterArg, %iterArg_0, %iterArg_1 : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT: }
