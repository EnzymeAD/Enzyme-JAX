// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s

func.func @sine(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sine %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @sine
// TUPLE: stablehlo.get_tuple_element
// TUPLE: stablehlo.get_tuple_element
// TUPLE-NOT: stablehlo.sine
// TUPLE: stablehlo.select
// TUPLE: stablehlo.convert
// TUPLE: stablehlo.select
// TUPLE: stablehlo.tuple

// FIRST-LABEL: func.func @sine
// FIRST: stablehlo.slice
// FIRST: stablehlo.slice
// FIRST-NOT: stablehlo.sine
// FIRST: stablehlo.select
// FIRST: stablehlo.convert
// FIRST: stablehlo.select
// FIRST: stablehlo.concatenate
