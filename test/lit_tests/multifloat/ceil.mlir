// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @ceil_test(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.ceil %0 : tensor<2xf64>
  return %1 : tensor<2xf64>
}

// FIRST-LABEL: func.func @ceil_test
// FIRST: %[[ADD:.*]] = stablehlo.add %[[HI_ARG0:.*]], %[[HI_ARG1:.*]] : tensor<1x2xf32>
// FIRST: %[[CH:.*]] = stablehlo.ceil %[[ADD]] : tensor<1x2xf32>
// FIRST: %[[IS_INT:.*]] = stablehlo.compare EQ, %[[CH]], %[[ADD]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// FIRST: %[[IS_POS:.*]] = stablehlo.compare GT, %[[LO_ADD:.*]], %[[ZERO]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST: %[[SHOULD_INC:.*]] = stablehlo.and %[[IS_INT]], %[[IS_POS]] : tensor<1x2xi1>
// FIRST: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1x2xf32>
// FIRST: %[[CH_PLUS_1:.*]] = stablehlo.add %[[CH]], %[[ONE]] : tensor<1x2xf32>
// FIRST: %[[RES_HI:.*]] = stablehlo.select %[[SHOULD_INC]], %[[CH_PLUS_1]], %[[CH]] : tensor<1x2xi1>, tensor<1x2xf32>
// FIRST: %[[ZERO_LO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// FIRST: %[[PACKED:.*]] = stablehlo.concatenate %[[RES_HI]], %[[ZERO_LO]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[CONV:.*]] = stablehlo.convert %[[PACKED]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[RES:.*]] = stablehlo.reduce(%[[CONV]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST: return %[[RES]] : tensor<2xf64>

// LAST-LABEL: func.func @ceil_test
// LAST: %[[ADD:.*]] = stablehlo.add %[[HI_ARG0:.*]], %[[HI_ARG1:.*]] : tensor<2x1xf32>
// LAST: %[[CH:.*]] = stablehlo.ceil %[[ADD]] : tensor<2x1xf32>
// LAST: %[[IS_INT:.*]] = stablehlo.compare EQ, %[[CH]], %[[ADD]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
// LAST: %[[IS_POS:.*]] = stablehlo.compare GT, %[[LO_ADD:.*]], %[[ZERO]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST: %[[SHOULD_INC:.*]] = stablehlo.and %[[IS_INT]], %[[IS_POS]] : tensor<2x1xi1>
// LAST: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2x1xf32>
// LAST: %[[CH_PLUS_1:.*]] = stablehlo.add %[[CH]], %[[ONE]] : tensor<2x1xf32>
// LAST: %[[RES_HI:.*]] = stablehlo.select %[[SHOULD_INC]], %[[CH_PLUS_1]], %[[CH]] : tensor<2x1xi1>, tensor<2x1xf32>
// LAST: %[[ZERO_LO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
// LAST: %[[PACKED:.*]] = stablehlo.concatenate %[[RES_HI]], %[[ZERO_LO]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[CONV:.*]] = stablehlo.convert %[[PACKED]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST: %[[RES:.*]] = stablehlo.reduce(%[[CONV]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST: return %[[RES]] : tensor<2xf64>

func.func @ceil_scalar(%arg0: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.ceil %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[2.1, 2.0]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[-0.05, -0.1]> : tensor<2xf64>
  
  %expected = stablehlo.constant dense<[3.0, 2.0]> : tensor<2xf64>
  
  %res = func.call @ceil_test(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %s1 = stablehlo.constant dense<1.500000e+00> : tensor<f64>
  %e1 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %r1 = func.call @ceil_scalar(%s1) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s2 = stablehlo.constant dense<-1.500000e+00> : tensor<f64>
  %e2 = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
  %r2 = func.call @ceil_scalar(%s2) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %e3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %r3 = func.call @ceil_scalar(%s3) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r3, %e3) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s4 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  %e4 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  %r4 = func.call @ceil_scalar(%s4) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r4, %e4) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
