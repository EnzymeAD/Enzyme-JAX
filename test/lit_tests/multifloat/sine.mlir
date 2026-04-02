// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s

func.func @sine(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sine %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @sine
// TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[SINE_HI:.*]] = stablehlo.sine %[[HI]] : tensor<2xf32>
// TUPLE: %[[SINE_LO:.*]] = stablehlo.sine %[[LO]] : tensor<2xf32>
// TUPLE: %{{.*}} = stablehlo.tuple %[[SINE_HI]], %[[SINE_LO]] : tuple<tensor<2xf32>, tensor<2xf32>>

// FIRST-LABEL: func.func @sine
// FIRST-NOT: stablehlo.get_tuple_element
// FIRST: stablehlo.sine
// FIRST-NOT: stablehlo.tuple
