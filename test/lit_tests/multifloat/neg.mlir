// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s

func.func @neg(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.negate %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @neg
// TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[NEG_HI:.*]] = stablehlo.negate %[[HI]] : tensor<2xf32>
// TUPLE: %[[NEG_LO:.*]] = stablehlo.negate %[[LO]] : tensor<2xf32>
// TUPLE: %{{.*}} = stablehlo.tuple %[[NEG_HI]], %[[NEG_LO]] : tuple<tensor<2xf32>, tensor<2xf32>>

// FIRST-LABEL: func.func @neg
// FIRST-NOT: stablehlo.get_tuple_element
// FIRST: stablehlo.negate
// FIRST-NOT: stablehlo.tuple
