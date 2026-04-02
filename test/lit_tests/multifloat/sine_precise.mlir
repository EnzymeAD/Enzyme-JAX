// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s

func.func @sine(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sine %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @sine
// TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE-NOT: stablehlo.sine %[[LO]]
// TUPLE: stablehlo.constant
// TUPLE: stablehlo.mul
// TUPLE: stablehlo.select
// TUPLE: stablehlo.convert
// TUPLE: stablehlo.select
// TUPLE: stablehlo.and
// TUPLE: stablehlo.select
