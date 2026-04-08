// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @exp_test(%arg0: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.exponential %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// FIRST-LABEL: func.func @exp_test
// FIRST: %[[HI:.*]] = stablehlo.slice %{{.*}} [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[LO:.*]] = stablehlo.slice %{{.*}} [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[LOG2E:.*]] = stablehlo.constant dense<1.44269502> : tensor<1xf32>
// FIRST: %[[MUL1:.*]] = stablehlo.multiply %[[HI]], %[[LOG2E]] : tensor<1xf32>
// FIRST: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<1xf32>
// FIRST: %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[HALF]] : tensor<1xf32>
// FIRST: %[[N:.*]] = stablehlo.floor %[[ADD1]] : tensor<1xf32>
// FIRST: %[[TWO:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<1xf32>
// FIRST: %[[SCALE:.*]] = stablehlo.power %[[TWO]], %{{.*}} : tensor<1xf32>

// LAST-LABEL: func.func @exp_test
// LAST: %[[HI:.*]] = stablehlo.slice %{{.*}} [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[LO:.*]] = stablehlo.slice %{{.*}} [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[LOG2E:.*]] = stablehlo.constant dense<1.44269502> : tensor<1xf32>
// LAST: %[[MUL1:.*]] = stablehlo.multiply %[[HI]], %[[LOG2E]] : tensor<1xf32>
// LAST: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<1xf32>
// LAST: %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[HALF]] : tensor<1xf32>
// LAST: %[[N:.*]] = stablehlo.floor %[[ADD1]] : tensor<1xf32>
// LAST: %[[TWO:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<1xf32>
// LAST: %[[SCALE:.*]] = stablehlo.power %[[TWO]], %{{.*}} : tensor<1xf32>

func.func @main() attributes {enzyme.no_multifloat} {
  %s1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  %e1 = stablehlo.constant dense<2.718281828459045e+00> : tensor<f64>
  %r1 = func.call @exp_test(%s1) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %e2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  %r2 = func.call @exp_test(%s2) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
