// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @floor_test(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.floor %0 : tensor<2xf64>
  return %1 : tensor<2xf64>
}

func.func @floor_scalar(%arg0: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.floor %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[2.1, 2.0]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[-0.05, -0.1]> : tensor<2xf64>
  
  %expected = stablehlo.constant dense<[2.0, 1.0]> : tensor<2xf64>
  
  %res = func.call @floor_test(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %s1 = stablehlo.constant dense<1.500000e+00> : tensor<f64>
  %e1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  %r1 = func.call @floor_scalar(%s1) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s2 = stablehlo.constant dense<-1.500000e+00> : tensor<f64>
  %e2 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  %r2 = func.call @floor_scalar(%s2) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %e3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %r3 = func.call @floor_scalar(%s3) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r3, %e3) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  %s4 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  %e4 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  %r4 = func.call @floor_scalar(%s4) : (tensor<f64>) -> tensor<f64>
  "check.expect_close"(%r4, %e4) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}

// FIRST: module {
// FIRST: func.func @floor_test(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// FIRST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// FIRST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2xf64>
// FIRST: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V15:.*]] = stablehlo.slice %[[V6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V17:.*]] = stablehlo.slice %[[V13]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[V18:.*]] = stablehlo.add %[[V14]], %[[V16]] : tensor<1x2xf32>
// FIRST: %[[V19:.*]] = stablehlo.subtract %[[V18]], %[[V16]] : tensor<1x2xf32>
// FIRST: %[[V20:.*]] = stablehlo.subtract %[[V18]], %[[V19]] : tensor<1x2xf32>
// FIRST: %[[V21:.*]] = stablehlo.subtract %[[V14]], %[[V19]] : tensor<1x2xf32>
// FIRST: %[[V22:.*]] = stablehlo.subtract %[[V16]], %[[V20]] : tensor<1x2xf32>
// FIRST: %[[V23:.*]] = stablehlo.add %[[V21]], %[[V22]] : tensor<1x2xf32>
// FIRST: %[[V24:.*]] = stablehlo.add %[[V15]], %[[V17]] : tensor<1x2xf32>
// FIRST: %[[V25:.*]] = stablehlo.subtract %[[V24]], %[[V17]] : tensor<1x2xf32>
// FIRST: %[[V26:.*]] = stablehlo.subtract %[[V24]], %[[V25]] : tensor<1x2xf32>
// FIRST: %[[V27:.*]] = stablehlo.subtract %[[V15]], %[[V25]] : tensor<1x2xf32>
// FIRST: %[[V28:.*]] = stablehlo.subtract %[[V17]], %[[V26]] : tensor<1x2xf32>
// FIRST: %[[V29:.*]] = stablehlo.add %[[V27]], %[[V28]] : tensor<1x2xf32>
// FIRST: %[[V30:.*]] = stablehlo.add %[[V18]], %[[V24]] : tensor<1x2xf32>
// FIRST: %[[V31:.*]] = stablehlo.subtract %[[V30]], %[[V18]] : tensor<1x2xf32>
// FIRST: %[[V32:.*]] = stablehlo.subtract %[[V24]], %[[V31]] : tensor<1x2xf32>
// FIRST: %[[V33:.*]] = stablehlo.add %[[V23]], %[[V29]] : tensor<1x2xf32>
// FIRST: %[[V34:.*]] = stablehlo.add %[[V33]], %[[V32]] : tensor<1x2xf32>
// FIRST: %[[V35:.*]] = stablehlo.add %[[V30]], %[[V34]] : tensor<1x2xf32>
// FIRST: %[[V36:.*]] = stablehlo.subtract %[[V35]], %[[V30]] : tensor<1x2xf32>
// FIRST: %[[V37:.*]] = stablehlo.subtract %[[V34]], %[[V36]] : tensor<1x2xf32>
// FIRST: %[[V38:.*]] = stablehlo.concatenate %[[V35]], %[[V37]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V39:.*]] = stablehlo.floor %[[V35]] : tensor<1x2xf32>
// FIRST: %[[V40:.*]] = stablehlo.subtract %[[V35]], %[[V39]] : tensor<1x2xf32>
// FIRST: %[[V41:.*]] = stablehlo.negate %[[V37]] : tensor<1x2xf32>
// FIRST: %[[V42:.*]] = stablehlo.compare LT, %[[V40]], %[[V41]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST: %[[Vcst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1x2xf32>
// FIRST: %[[V43:.*]] = stablehlo.subtract %[[V39]], %[[Vcst]] : tensor<1x2xf32>
// FIRST: %[[V44:.*]] = stablehlo.select %[[V42]], %[[V43]], %[[V39]] : tensor<1x2xi1>, tensor<1x2xf32>
// FIRST: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// FIRST: %[[V45:.*]] = stablehlo.concatenate %[[V44]], %[[Vcst_0]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[V46:.*]] = stablehlo.convert %[[V45]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[V47:.*]] = stablehlo.reduce(%[[V46]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST: return %[[V47]] : tensor<2xf64>
// FIRST: }
// FIRST: func.func @floor_scalar(%arg0: tensor<f64>) -> tensor<f64> {
// FIRST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// FIRST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<f32>) -> tensor<f64>
// FIRST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<f64>
// FIRST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<f64>) -> tensor<f32>
// FIRST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<f32>) -> tensor<1xf32>
// FIRST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<f32>) -> tensor<1xf32>
// FIRST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST: %[[V7:.*]] = stablehlo.slice %[[V6]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V8:.*]] = stablehlo.slice %[[V6]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// FIRST: %[[V9:.*]] = stablehlo.floor %[[V7]] : tensor<1xf32>
// FIRST: %[[V10:.*]] = stablehlo.subtract %[[V7]], %[[V9]] : tensor<1xf32>
// FIRST: %[[V11:.*]] = stablehlo.negate %[[V8]] : tensor<1xf32>
// FIRST: %[[V12:.*]] = stablehlo.compare LT, %[[V10]], %[[V11]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// FIRST: %[[Vcst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
// FIRST: %[[V13:.*]] = stablehlo.subtract %[[V9]], %[[Vcst]] : tensor<1xf32>
// FIRST: %[[V14:.*]] = stablehlo.select %[[V12]], %[[V13]], %[[V9]] : tensor<1xi1>, tensor<1xf32>
// FIRST: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// FIRST: %[[V15:.*]] = stablehlo.concatenate %[[V14]], %[[Vcst_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST: %[[V16:.*]] = stablehlo.convert %[[V15]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[V17:.*]] = stablehlo.reduce(%[[V16]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// FIRST: return %[[V17]] : tensor<f64>
// FIRST: }

// LAST: module {
// LAST: func.func @floor_test(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// LAST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// LAST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V7:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[V9:.*]] = stablehlo.subtract %arg1, %[[V8]] : tensor<2xf64>
// LAST: %[[V10:.*]] = stablehlo.convert %[[V9]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[V11:.*]] = stablehlo.reshape %[[V7]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[V13:.*]] = stablehlo.concatenate %[[V11]], %[[V12]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V14:.*]] = stablehlo.slice %[[V6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V15:.*]] = stablehlo.slice %[[V6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V16:.*]] = stablehlo.slice %[[V13]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V17:.*]] = stablehlo.slice %[[V13]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[V18:.*]] = stablehlo.add %[[V14]], %[[V16]] : tensor<2x1xf32>
// LAST: %[[V19:.*]] = stablehlo.subtract %[[V18]], %[[V16]] : tensor<2x1xf32>
// LAST: %[[V20:.*]] = stablehlo.subtract %[[V18]], %[[V19]] : tensor<2x1xf32>
// LAST: %[[V21:.*]] = stablehlo.subtract %[[V14]], %[[V19]] : tensor<2x1xf32>
// LAST: %[[V22:.*]] = stablehlo.subtract %[[V16]], %[[V20]] : tensor<2x1xf32>
// LAST: %[[V23:.*]] = stablehlo.add %[[V21]], %[[V22]] : tensor<2x1xf32>
// LAST: %[[V24:.*]] = stablehlo.add %[[V15]], %[[V17]] : tensor<2x1xf32>
// LAST: %[[V25:.*]] = stablehlo.subtract %[[V24]], %[[V17]] : tensor<2x1xf32>
// LAST: %[[V26:.*]] = stablehlo.subtract %[[V24]], %[[V25]] : tensor<2x1xf32>
// LAST: %[[V27:.*]] = stablehlo.subtract %[[V15]], %[[V25]] : tensor<2x1xf32>
// LAST: %[[V28:.*]] = stablehlo.subtract %[[V17]], %[[V26]] : tensor<2x1xf32>
// LAST: %[[V29:.*]] = stablehlo.add %[[V27]], %[[V28]] : tensor<2x1xf32>
// LAST: %[[V30:.*]] = stablehlo.add %[[V18]], %[[V24]] : tensor<2x1xf32>
// LAST: %[[V31:.*]] = stablehlo.subtract %[[V30]], %[[V18]] : tensor<2x1xf32>
// LAST: %[[V32:.*]] = stablehlo.subtract %[[V24]], %[[V31]] : tensor<2x1xf32>
// LAST: %[[V33:.*]] = stablehlo.add %[[V23]], %[[V29]] : tensor<2x1xf32>
// LAST: %[[V34:.*]] = stablehlo.add %[[V33]], %[[V32]] : tensor<2x1xf32>
// LAST: %[[V35:.*]] = stablehlo.add %[[V30]], %[[V34]] : tensor<2x1xf32>
// LAST: %[[V36:.*]] = stablehlo.subtract %[[V35]], %[[V30]] : tensor<2x1xf32>
// LAST: %[[V37:.*]] = stablehlo.subtract %[[V34]], %[[V36]] : tensor<2x1xf32>
// LAST: %[[V38:.*]] = stablehlo.concatenate %[[V35]], %[[V37]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V39:.*]] = stablehlo.floor %[[V35]] : tensor<2x1xf32>
// LAST: %[[V40:.*]] = stablehlo.subtract %[[V35]], %[[V39]] : tensor<2x1xf32>
// LAST: %[[V41:.*]] = stablehlo.negate %[[V37]] : tensor<2x1xf32>
// LAST: %[[V42:.*]] = stablehlo.compare LT, %[[V40]], %[[V41]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST: %[[Vcst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2x1xf32>
// LAST: %[[V43:.*]] = stablehlo.subtract %[[V39]], %[[Vcst]] : tensor<2x1xf32>
// LAST: %[[V44:.*]] = stablehlo.select %[[V42]], %[[V43]], %[[V39]] : tensor<2x1xi1>, tensor<2x1xf32>
// LAST: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
// LAST: %[[V45:.*]] = stablehlo.concatenate %[[V44]], %[[Vcst_0]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[V46:.*]] = stablehlo.convert %[[V45]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST: %[[V47:.*]] = stablehlo.reduce(%[[V46]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST: return %[[V47]] : tensor<2xf64>
// LAST: }
// LAST: func.func @floor_scalar(%arg0: tensor<f64>) -> tensor<f64> {
// LAST: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// LAST: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<f32>) -> tensor<f64>
// LAST: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<f64>
// LAST: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<f64>) -> tensor<f32>
// LAST: %[[V4:.*]] = stablehlo.reshape %[[V0]] : (tensor<f32>) -> tensor<1xf32>
// LAST: %[[V5:.*]] = stablehlo.reshape %[[V3]] : (tensor<f32>) -> tensor<1xf32>
// LAST: %[[V6:.*]] = stablehlo.concatenate %[[V4]], %[[V5]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST: %[[V7:.*]] = stablehlo.slice %[[V6]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V8:.*]] = stablehlo.slice %[[V6]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// LAST: %[[V9:.*]] = stablehlo.floor %[[V7]] : tensor<1xf32>
// LAST: %[[V10:.*]] = stablehlo.subtract %[[V7]], %[[V9]] : tensor<1xf32>
// LAST: %[[V11:.*]] = stablehlo.negate %[[V8]] : tensor<1xf32>
// LAST: %[[V12:.*]] = stablehlo.compare LT, %[[V10]], %[[V11]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// LAST: %[[Vcst:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
// LAST: %[[V13:.*]] = stablehlo.subtract %[[V9]], %[[Vcst]] : tensor<1xf32>
// LAST: %[[V14:.*]] = stablehlo.select %[[V12]], %[[V13]], %[[V9]] : tensor<1xi1>, tensor<1xf32>
// LAST: %[[Vcst_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// LAST: %[[V15:.*]] = stablehlo.concatenate %[[V14]], %[[Vcst_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST: %[[V16:.*]] = stablehlo.convert %[[V15]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[Vcst_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST: %[[V17:.*]] = stablehlo.reduce(%[[V16]] init: %[[Vcst_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// LAST: return %[[V17]] : tensor<f64>
// LAST: }
