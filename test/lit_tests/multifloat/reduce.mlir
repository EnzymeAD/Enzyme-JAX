// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE_LIMB %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple precise-reduce=true" %s | FileCheck --check-prefix=TUPLE_PRECISE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @reduce_test(%arg0: tensor<2x2xf64>) -> tensor<2xf64> {
  %cst = stablehlo.constant dense<0.0> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// FIRST_LIMB-LABEL: func.func @reduce_test
// FIRST_LIMB: stablehlo.reduce
// FIRST_LIMB: stablehlo.reduce

// LAST_LIMB-LABEL: func.func @reduce_test
// LAST_LIMB: stablehlo.reduce
// LAST_LIMB: stablehlo.reduce

// TUPLE_LIMB-LABEL: func.func @reduce_test
// TUPLE_LIMB: stablehlo.reduce
// TUPLE_LIMB: stablehlo.reduce

// TUPLE_PRECISE-LABEL: func.func @reduce_test
// TUPLE_PRECISE: stablehlo.reduce({{.*}}, {{.*}} init: {{.*}}, {{.*}})

func.func @main() attributes {enzyme.no_multifloat} {
  %c = stablehlo.constant dense<[[1.1, 2.2], [3.3, 4.4]]> : tensor<2x2xf64>
  %expected = stablehlo.constant dense<[4.4, 6.6]> : tensor<2xf64>
  
  %res = func.call @reduce_test(%c) : (tensor<2x2xf64>) -> tensor<2xf64>
  
  "check.expect_close"(%res, %expected) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
