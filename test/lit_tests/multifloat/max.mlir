// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple dot-general-to-reduce=false" %s | FileCheck %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple dot-general-to-reduce=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_max(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_max
// CHECK: %[[V0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[V1:.*]] = stablehlo.convert %[[V0]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[V2:.*]] = stablehlo.subtract %arg0, %[[V1]] : tensor<2xf64>
// CHECK: %[[V3:.*]] = stablehlo.convert %[[V2]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[V4:.*]] = stablehlo.tuple %[[V0]], %[[V3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// CHECK: %[[V5:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[V6:.*]] = stablehlo.convert %[[V5]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[V7:.*]] = stablehlo.subtract %arg1, %[[V6]] : tensor<2xf64>
// CHECK: %[[V8:.*]] = stablehlo.convert %[[V7]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK: %[[V9:.*]] = stablehlo.tuple %[[V5]], %[[V8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// CHECK: %[[V10:.*]] = stablehlo.get_tuple_element %[[V4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V11:.*]] = stablehlo.get_tuple_element %[[V4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V12:.*]] = stablehlo.get_tuple_element %[[V9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V13:.*]] = stablehlo.get_tuple_element %[[V9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V14:.*]] = stablehlo.compare GT, %[[V10]], %[[V12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK: %[[V15:.*]] = stablehlo.compare EQ, %[[V10]], %[[V12]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK: %[[V16:.*]] = stablehlo.compare GT, %[[V11]], %[[V13]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK: %[[V17:.*]] = stablehlo.select %[[V15]], %[[V16]], %[[V14]] : tensor<2xi1>, tensor<2xi1>
// CHECK: %[[V18:.*]] = stablehlo.get_tuple_element %[[V4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V19:.*]] = stablehlo.get_tuple_element %[[V4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V20:.*]] = stablehlo.get_tuple_element %[[V9]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V21:.*]] = stablehlo.get_tuple_element %[[V9]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// CHECK: %[[V22:.*]] = stablehlo.select %[[V17]], %[[V18]], %[[V20]] : tensor<2xi1>, tensor<2xf32>
// CHECK: %[[V23:.*]] = stablehlo.select %[[V17]], %[[V19]], %[[V21]] : tensor<2xi1>, tensor<2xf32>
// CHECK: %[[V24:.*]] = stablehlo.tuple %[[V22]], %[[V23]] : tuple<tensor<2xf32>, tensor<2xf32>>
// CHECK: %[[V25:.*]] = stablehlo.convert %[[V22]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[V26:.*]] = stablehlo.convert %[[V23]] : (tensor<2xf32>) -> tensor<2xf64>
// CHECK: %[[V27:.*]] = stablehlo.add %[[V25]], %[[V26]] : tensor<2xf64>
// CHECK: return %[[V27]] : tensor<2xf64>

func.func @test_reduce_max(%arg0: tensor<2x2xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<-1.000000e+30> : tensor<f64>
  %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<f64>
  return %0 : tensor<f64>
}

// CHECK-LABEL: func.func @test_reduce_max
// CHECK: %0 = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xf32>
// CHECK: %[[V8:.*]]:2 = stablehlo.reduce(%{{.*}} init: %[[Vcst:.*]]), (%{{.*}} init: %[[Vcst_0:.*]]) across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
// CHECK: reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<f32>, %arg4: tensor<f32>)  {
// CHECK: %[[V13:.*]] = stablehlo.compare GT, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[V14:.*]] = stablehlo.compare EQ, %arg1, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[V15:.*]] = stablehlo.compare GT, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK: %[[V16:.*]] = stablehlo.select %[[V14]], %[[V15]], %[[V13]] : tensor<i1>, tensor<i1>
// CHECK: %[[V17:.*]] = stablehlo.select %[[V16]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK: %[[V18:.*]] = stablehlo.select %[[V16]], %arg2, %arg4 : tensor<i1>, tensor<f32>
// CHECK: stablehlo.return %[[V17]], %[[V18]] : tensor<f32>, tensor<f32>
// CHECK: }
// CHECK: %[[V10:.*]] = stablehlo.convert %[[V8]]#0 : (tensor<f32>) -> tensor<f64>
// CHECK: %[[V11:.*]] = stablehlo.convert %[[V8]]#1 : (tensor<f32>) -> tensor<f64>
// CHECK: %[[V12:.*]] = stablehlo.add %[[V10]], %[[V11]] : tensor<f64>
// CHECK: return %[[V12]] : tensor<f64>

func.func @main() attributes {enzyme.no_multifloat} {
  %c1 = stablehlo.constant dense<[1.5, 2.0]> : tensor<2xf64>
  %c2 = stablehlo.constant dense<[1.0, 3.0]> : tensor<2xf64>
  %expected_max = stablehlo.constant dense<[1.5, 3.0]> : tensor<2xf64>
  
  %res_max = func.call @test_max(%c1, %c2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  "check.expect_close"(%res_max, %expected_max) {max_ulp_difference = 100 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  %m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %expected_reduce = stablehlo.constant dense<4.0> : tensor<f64>
  
  %res_reduce = func.call @test_reduce_max(%m) : (tensor<2x2xf64>) -> tensor<f64>
  "check.expect_close"(%res_reduce, %expected_reduce) {max_ulp_difference = 100 : ui64} : (tensor<f64>, tensor<f64>) -> ()
  
  return
}
