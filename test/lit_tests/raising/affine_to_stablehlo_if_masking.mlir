// RUN: enzymexlamlir-opt %s -raise-affine-to-stablehlo | FileCheck %s

func.func @test_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

func.func @test_nested_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>, %arg2: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond1 = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond1 {
      %cond2 = affine.load %arg2[%i] : memref<10xi1>
      scf.if %cond2 {
        %v = affine.load %arg0[%i] : memref<10xf32>
        %v2 = arith.addf %v, %v : f32
        affine.store %v2, %arg0[%i] : memref<10xf32>
      }
    }
  }
  return
}

func.func @test_if_else_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    } else {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.mulf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

func.func @test_affine_if_masking(%arg0: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL:   func.func private @test_affine_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK:           %[[V0:.*]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[V1:.*]] = stablehlo.add %[[V0]], %[[C0]] : tensor<10xi64>
// CHECK:           %[[C1:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[V2:.*]] = stablehlo.multiply %[[V1]], %[[C1]] : tensor<10xi64>
// CHECK:           %[[CM5:.*]] = stablehlo.constant dense<-5> : tensor<i64>
// CHECK:           %[[V3:.*]] = stablehlo.broadcast_in_dim %[[CM5]], dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK:           %[[V4:.*]] = stablehlo.add %[[V2]], %[[V3]] : tensor<10xi64>
// CHECK:           %[[C0B:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[V5:.*]] = stablehlo.compare  GE, %[[V4]], %[[C0B]] : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK:           %[[C0C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[V6:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0C]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[V7:.*]] = stablehlo.reshape %[[V6]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V8:.*]] = arith.addf %[[V7]], %[[V7]] : tensor<10xf32>
// CHECK:           %[[C0D:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[V9:.*]] = stablehlo.broadcast_in_dim %[[V8]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V10:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0D]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[V11:.*]] = stablehlo.reshape %[[V9]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V12:.*]] = stablehlo.reshape %[[V10]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V13:.*]] = stablehlo.select %[[V5]], %[[V11]], %[[V12]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[V14:.*]] = stablehlo.reshape %[[V13]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V15:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[V14]], %[[C0D]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           return %[[V15]] : tensor<10xf32>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_if_else_masking_raised(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[C10:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %[[W0:.*]]:3 = stablehlo.while(%iterArg = %[[C0]], %iterArg_2 = %[[VAL_0]], %iterArg_3 = %[[VAL_1]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           cond {
// CHECK:           %[[WCOND:.*]] = stablehlo.compare  LT, %iterArg, %[[C10]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[WCOND]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[DS1:.*]] = stablehlo.dynamic_slice %iterArg_3, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[COND:.*]] = stablehlo.reshape %[[DS1]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %[[DS2:.*]] = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[V1:.*]] = stablehlo.reshape %[[DS2]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[V2:.*]] = arith.addf %[[V1]], %[[V1]] : tensor<f32>
// CHECK:           %[[BID1:.*]] = stablehlo.broadcast_in_dim %[[V2]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[DS3:.*]] = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[RSH1:.*]] = stablehlo.reshape %[[BID1]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[RSH2:.*]] = stablehlo.reshape %[[DS3]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[BID2:.*]] = stablehlo.broadcast_in_dim %[[COND]], dims = [] : (tensor<i1>) -> tensor<1xi1>
// CHECK:           %[[SEL1:.*]] = stablehlo.select %[[BID2]], %[[RSH1]], %[[RSH2]] : tensor<1xi1>, tensor<1xf32>
// CHECK:           %[[RSH3:.*]] = stablehlo.reshape %[[SEL1]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[DUS1:.*]] = stablehlo.dynamic_update_slice %iterArg_2, %[[RSH3]], %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[NOT:.*]] = stablehlo.not %[[COND]] : tensor<i1>
// CHECK:           %[[DS4:.*]] = stablehlo.dynamic_slice %[[DUS1]], %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[V3:.*]] = stablehlo.reshape %[[DS4]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[V4:.*]] = arith.mulf %[[V3]], %[[V3]] : tensor<f32>
// CHECK:           %[[BID3:.*]] = stablehlo.broadcast_in_dim %[[V4]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[DS5:.*]] = stablehlo.dynamic_slice %[[DUS1]], %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[RSH4:.*]] = stablehlo.reshape %[[BID3]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[RSH5:.*]] = stablehlo.reshape %[[DS5]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[BID4:.*]] = stablehlo.broadcast_in_dim %[[NOT]], dims = [] : (tensor<i1>) -> tensor<1xi1>
// CHECK:           %[[SEL2:.*]] = stablehlo.select %[[BID4]], %[[RSH4]], %[[RSH5]] : tensor<1xi1>, tensor<1xf32>
// CHECK:           %[[RSH6:.*]] = stablehlo.reshape %[[SEL2]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[DUS2:.*]] = stablehlo.dynamic_update_slice %[[DUS1]], %[[RSH6]], %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[INC:.*]] = stablehlo.add %iterArg, %[[C1]] : tensor<i64>
// CHECK:           stablehlo.return %[[INC]], %[[DUS2]], %iterArg_3 : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           }
// CHECK:           return %[[W0]]#1, %[[W0]]#2 : tensor<10xf32>, tensor<10xi1>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_nested_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>, %[[VAL_2:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>, tensor<10xi1>) {
// CHECK:           %[[V0:.*]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[V1:.*]] = stablehlo.add %[[V0]], %[[C0]] : tensor<10xi64>
// CHECK:           %[[C1:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[V2:.*]] = stablehlo.multiply %[[V1]], %[[C1]] : tensor<10xi64>
// CHECK:           %[[C0B:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[DS1:.*]] = stablehlo.dynamic_slice %[[VAL_1]], %[[C0B]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[V3:.*]] = stablehlo.reshape %[[DS1]] : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[C0C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[DS2:.*]] = stablehlo.dynamic_slice %[[VAL_2]], %[[C0C]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[V4:.*]] = stablehlo.reshape %[[DS2]] : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[COND:.*]] = stablehlo.and %[[V3]], %[[V4]] : tensor<10xi1>
// CHECK:           %[[C0D:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[DS3:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0D]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[V5:.*]] = stablehlo.reshape %[[DS3]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V6:.*]] = arith.addf %[[V5]], %[[V5]] : tensor<10xf32>
// CHECK:           %[[C0E:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[BID1:.*]] = stablehlo.broadcast_in_dim %[[V6]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[DS4:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0E]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[RSH1:.*]] = stablehlo.reshape %[[BID1]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[RSH2:.*]] = stablehlo.reshape %[[DS4]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[COND]], %[[RSH1]], %[[RSH2]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[RSH3:.*]] = stablehlo.reshape %[[SEL]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[RSH3]], %[[C0E]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           return %[[DUS]], %[[VAL_1]], %[[VAL_2]] : tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_if_masking_raised(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %[[V0:.*]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[V1:.*]] = stablehlo.add %[[V0]], %[[C0]] : tensor<10xi64>
// CHECK:           %[[C1:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[V2:.*]] = stablehlo.multiply %[[V1]], %[[C1]] : tensor<10xi64>
// CHECK:           %[[C0B:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[DS1:.*]] = stablehlo.dynamic_slice %[[VAL_1]], %[[C0B]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[V3:.*]] = stablehlo.reshape %[[DS1]] : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[C0C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[DS2:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0C]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[V4:.*]] = stablehlo.reshape %[[DS2]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[V5:.*]] = arith.addf %[[V4]], %[[V4]] : tensor<10xf32>
// CHECK:           %[[C0D:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[BID1:.*]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[DS3:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[C0D]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[RSH1:.*]] = stablehlo.reshape %[[BID1]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[RSH2:.*]] = stablehlo.reshape %[[DS3]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[V3]], %[[RSH1]], %[[RSH2]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[RSH3:.*]] = stablehlo.reshape %[[SEL]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[RSH3]], %[[C0D]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           return %[[DUS]], %[[VAL_1]] : tensor<10xf32>, tensor<10xi1>
// CHECK:         }
